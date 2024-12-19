use std::collections::HashSet;

use crate::{
    const_expr::reduce_const_expr, parser::TypeSyntax, scope::SymbolScope,
    source_ref::SourceRefSliceEq, spirv as spv, symbol::meta::SymbolAttribute, utils::roundup2,
};

mod conversion_rules;
pub use self::conversion_rules::*;

/// 組み込みスカラ型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicScalarType {
    /// void
    Unit,
    /// ブール
    Bool,
    /// 符号なし32bit整数
    UInt,
    /// 符号付き32bit整数
    SInt,
    /// 32bit浮動小数点数
    Float,
    /// 型が明示されていない整数（リテラルなど）
    UnknownIntClass,
    /// 型が明示されていない実数（リテラルなど）
    UnknownNumberClass,
}
impl IntrinsicScalarType {
    pub const fn vec(self, count: u8) -> IntrinsicVectorType {
        IntrinsicVectorType(self, count)
    }

    #[inline]
    pub fn make_spv_type(&self) -> spv::Type {
        match self {
            Self::Unit => spv::Type::Void,
            Self::Bool => spv::ScalarType::Bool.into(),
            Self::UInt => spv::ScalarType::Int(32, false).into(),
            Self::SInt => spv::ScalarType::Int(32, true).into(),
            Self::Float => spv::ScalarType::Float(32).into(),
            Self::UnknownIntClass | Self::UnknownNumberClass => {
                unreachable!("unknown type classes left unresolved")
            }
        }
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            // Unknown Value Classes are treated as sint/float
            Self::UnknownIntClass | Self::UnknownNumberClass => Some(4),
        }
    }

    pub const fn std140_size(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            // Unknown Value Classes are treated as sint/float
            Self::UnknownIntClass | Self::UnknownNumberClass => Some(4),
        }
    }

    pub const fn is_unknown_type(&self) -> bool {
        matches!(self, Self::UnknownIntClass | Self::UnknownNumberClass)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntrinsicVectorType(pub IntrinsicScalarType, pub u8);
impl IntrinsicVectorType {
    pub const fn with_type(self, ty: IntrinsicScalarType) -> Self {
        Self(ty, self.1)
    }

    pub const fn with_len(self, len: u8) -> Self {
        Self(self.0, len)
    }

    pub const fn scalar(&self) -> IntrinsicScalarType {
        self.0
    }

    pub const fn len(&self) -> u8 {
        self.1
    }

    #[inline]
    pub fn make_spv_type(&self) -> spv::Type {
        if self.1 < 2 {
            unreachable!("too short vector generated");
        }

        if self.1 > 4 {
            unreachable!("too big vector generated");
        }

        self.0.make_spv_type().of_vector(self.1 as _)
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self(IntrinsicScalarType::Unit, _) => None,
            Self(s, 1) => s.std140_alignment(),
            Self(_, 2) => Some(8),
            // vec3 has same alignment as vec4
            Self(_, 3) | Self(_, 4) => Some(16),
            _ => None,
        }
    }

    pub const fn std140_size(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self(IntrinsicScalarType::Unit, _) => None,
            Self(s, 1) => s.std140_alignment(),
            Self(_, 2) => Some(8),
            Self(_, 3) => Some(12),
            Self(_, 4) => Some(16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntrinsicMatrixType(pub IntrinsicVectorType, pub u8);
impl IntrinsicMatrixType {
    pub const fn with_scalar_type(self, new_st: IntrinsicScalarType) -> Self {
        Self(self.0.with_type(new_st), self.1)
    }

    pub fn make_spv_type(&self) -> spv::Type {
        if self.1 < 2 {
            unreachable!("too short matrix type");
        }

        if self.1 > 4 {
            unreachable!("too big matrix type");
        }

        self.0.make_spv_type().vector_to_matrix(self.1 as _)
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            Self(IntrinsicVectorType(IntrinsicScalarType::Float, 2), _) => Some(8),
            Self(IntrinsicVectorType(IntrinsicScalarType::Float, _), _) => Some(16),
            _ => None,
        }
    }

    pub const fn std140_size(&self) -> Option<usize> {
        Some(16)
    }
}

/// 組み込みの複合データ型のクラス分類
pub enum IntrinsicCompositeTypeClass {
    /// スカラ
    Scalar,
    /// ベクトル
    Vector(u8),
    /// 行列
    Matrix(u8, u8),
}
impl IntrinsicCompositeTypeClass {
    /// スカラ型と組み合わせて具体的な型を作る
    pub const fn combine_scalar(self, scalar: IntrinsicScalarType) -> IntrinsicType {
        match self {
            Self::Scalar => IntrinsicType::Scalar(scalar),
            Self::Vector(c) => IntrinsicType::Vector(IntrinsicVectorType(scalar, c)),
            Self::Matrix(c, r) => {
                IntrinsicType::Matrix(IntrinsicMatrixType(IntrinsicVectorType(scalar, c), r))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicType {
    Scalar(IntrinsicScalarType),
    Vector(IntrinsicVectorType),
    Matrix(IntrinsicMatrixType),
    Texture1D,
    Texture2D,
    Texture3D,
    Image1D,
    Image2D,
    Image3D,
    SubpassInput,
}
impl IntrinsicType {
    pub const UNIT: Self = Self::Scalar(IntrinsicScalarType::Unit);

    pub const fn bvec(component_count: u8) -> Self {
        Self::Vector(IntrinsicScalarType::Bool.vec(component_count))
    }

    pub const fn uvec(component_count: u8) -> Self {
        Self::Vector(IntrinsicScalarType::UInt.vec(component_count))
    }

    pub const fn ivec(component_count: u8) -> Self {
        Self::Vector(IntrinsicScalarType::SInt.vec(component_count))
    }

    pub const fn vec(component_count: u8) -> Self {
        Self::Vector(IntrinsicScalarType::Float.vec(component_count))
    }

    pub const fn mat(row_count: u8, column_count: u8) -> Self {
        Self::Matrix(IntrinsicMatrixType(
            IntrinsicVectorType(IntrinsicScalarType::Float, row_count),
            column_count,
        ))
    }

    pub const fn try_cast_scalar(self, target: IntrinsicScalarType) -> Option<Self> {
        match self {
            Self::Scalar(_) => Some(Self::Scalar(target)),
            Self::Vector(v) => Some(Self::Vector(v.with_type(target))),
            Self::Matrix(m) => Some(Self::Matrix(m.with_scalar_type(target))),
            _ => None,
        }
    }

    pub const fn only_scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            &Self::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            &Self::Scalar(s) => Some(s),
            &Self::Vector(IntrinsicVectorType(s, _)) => Some(s),
            &Self::Matrix(IntrinsicMatrixType(IntrinsicVectorType(s, _), _)) => Some(s),
            _ => None,
        }
    }

    pub const fn composite_type_class(&self) -> Option<IntrinsicCompositeTypeClass> {
        match self {
            Self::Scalar(_) => Some(IntrinsicCompositeTypeClass::Scalar),
            Self::Vector(IntrinsicVectorType(_, c)) => {
                Some(IntrinsicCompositeTypeClass::Vector(*c))
            }
            Self::Matrix(IntrinsicMatrixType(IntrinsicVectorType(_, c), r)) => {
                Some(IntrinsicCompositeTypeClass::Matrix(*c, *r))
            }
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            &Self::Scalar(_) => Some(1),
            &Self::Vector(IntrinsicVectorType(_, e)) => Some(e),
            _ => None,
        }
    }

    pub const fn can_uniform_struct_member(&self) -> bool {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Scalar(IntrinsicScalarType::Unit) => false,
            // samplers/image refs cannot be a member of uniform struct
            Self::Image1D
            | Self::Image2D
            | Self::Image3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => false,
            _ => true,
        }
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            Self::Scalar(s) => s.std140_alignment(),
            Self::Vector(s) => s.std140_alignment(),
            Self::Matrix(s) => s.std140_alignment(),
            // samplers/image refs cannot be a member of uniform struct
            Self::Image1D
            | Self::Image2D
            | Self::Image3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => None,
        }
    }

    pub const fn std140_size(&self) -> Option<usize> {
        match self {
            Self::Scalar(s) => s.std140_size(),
            Self::Vector(s) => s.std140_size(),
            Self::Matrix(s) => s.std140_size(),
            // samplers/image refs cannot be a member of uniform struct
            Self::Image1D
            | Self::Image2D
            | Self::Image3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => None,
        }
    }

    pub const fn is_scalar_type(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    pub const fn is_vector_type(&self) -> bool {
        matches!(self, Self::Vector(_))
    }

    pub const fn is_matrix_type(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }

    pub fn make_spv_type(&self) -> spv::Type {
        match self {
            Self::Scalar(s) => s.make_spv_type(),
            Self::Vector(s) => s.make_spv_type(),
            Self::Matrix(s) => s.make_spv_type(),
            Self::Image1D => unimplemented!(),
            Self::Image2D => unimplemented!(),
            Self::Image3D => unimplemented!(),
            Self::Texture1D => unimplemented!(),
            Self::Texture2D => unimplemented!(),
            Self::Texture3D => unimplemented!(),
            Self::SubpassInput => spv::Type::SUBPASS_DATA_IMAGE_TYPE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefStorageClass {
    Local,
    Input,
    Uniform,
    Device,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConcreteType<'s> {
    Generic(Vec<usize>, Box<ConcreteType<'s>>),
    GenericVar(usize),
    Intrinsic(IntrinsicType),
    UserDefined {
        name: &'s str,
        generic_args: Vec<ConcreteType<'s>>,
    },
    Struct(Vec<UserDefinedStructMember<'s>>),
    Tuple(Vec<ConcreteType<'s>>),
    Array(Box<ConcreteType<'s>>, u32),
    Function {
        args: Vec<ConcreteType<'s>>,
        output: Option<Box<ConcreteType<'s>>>,
    },
    IntrinsicTypeConstructor(IntrinsicType),
    OverloadedFunctions(Vec<(Vec<ConcreteType<'s>>, Box<ConcreteType<'s>>)>),
    Ref(Box<ConcreteType<'s>>, RefStorageClass),
    MutableRef(Box<ConcreteType<'s>>, RefStorageClass),
    Never,
}
impl<'s> ConcreteType<'s> {
    pub fn build(
        symbol_scope: &SymbolScope<'_, 's>,
        sibling_scope_opaque_symbols: &HashSet<&'s str>,
        t: TypeSyntax<'s>,
    ) -> Self {
        match t {
            TypeSyntax::Simple {
                name_token,
                generic_args,
            } => match name_token.slice {
                "UInt" => Self::Intrinsic(IntrinsicType::Scalar(IntrinsicScalarType::UInt)),
                "UInt2" => Self::Intrinsic(IntrinsicType::uvec(2)),
                "UInt3" => Self::Intrinsic(IntrinsicType::uvec(3)),
                "UInt4" => Self::Intrinsic(IntrinsicType::uvec(4)),
                "SInt" | "Int" => Self::Intrinsic(IntrinsicType::Scalar(IntrinsicScalarType::SInt)),
                "SInt2" | "Int2" => Self::Intrinsic(IntrinsicType::ivec(2)),
                "SInt3" | "Int3" => Self::Intrinsic(IntrinsicType::ivec(3)),
                "SInt4" | "Int4" => Self::Intrinsic(IntrinsicType::ivec(4)),
                "Float" => Self::Intrinsic(IntrinsicType::Scalar(IntrinsicScalarType::Float)),
                "Float2" => Self::Intrinsic(IntrinsicType::vec(2)),
                "Float3" => Self::Intrinsic(IntrinsicType::vec(3)),
                "Float4" => Self::Intrinsic(IntrinsicType::vec(4)),
                "Float2x2" => Self::Intrinsic(IntrinsicType::mat(2, 2)),
                "Float2x3" => Self::Intrinsic(IntrinsicType::mat(2, 3)),
                "Float2x4" => Self::Intrinsic(IntrinsicType::mat(2, 4)),
                "Float3x2" => Self::Intrinsic(IntrinsicType::mat(3, 2)),
                "Float3x3" => Self::Intrinsic(IntrinsicType::mat(3, 3)),
                "Float3x4" => Self::Intrinsic(IntrinsicType::mat(3, 4)),
                "Float4x2" => Self::Intrinsic(IntrinsicType::mat(4, 2)),
                "Float4x3" => Self::Intrinsic(IntrinsicType::mat(4, 3)),
                "Float4x4" => Self::Intrinsic(IntrinsicType::mat(4, 4)),
                "Image1D" => Self::Intrinsic(IntrinsicType::Image1D),
                "Image2D" => Self::Intrinsic(IntrinsicType::Image2D),
                "Image3D" => Self::Intrinsic(IntrinsicType::Image3D),
                "Texture1D" => Self::Intrinsic(IntrinsicType::Texture1D),
                "Texture2D" => Self::Intrinsic(IntrinsicType::Texture2D),
                "Texture3D" => Self::Intrinsic(IntrinsicType::Texture3D),
                "SubpassInput" => Self::Intrinsic(IntrinsicType::SubpassInput),
                name => {
                    if sibling_scope_opaque_symbols.contains(name) {
                        ConcreteType::UserDefined {
                            name: name_token.slice,
                            generic_args: generic_args
                                .map_or_else(Vec::new, |x| x.args)
                                .into_iter()
                                .map(|x| {
                                    ConcreteType::build(
                                        symbol_scope,
                                        sibling_scope_opaque_symbols,
                                        x.0,
                                    )
                                })
                                .collect(),
                        }
                    } else {
                        match symbol_scope.lookup_user_defined_type(name) {
                            Some(_) => ConcreteType::UserDefined {
                                name: name_token.slice,
                                generic_args: generic_args
                                    .map_or_else(Vec::new, |x| x.args)
                                    .into_iter()
                                    .map(|x| {
                                        ConcreteType::build(
                                            symbol_scope,
                                            sibling_scope_opaque_symbols,
                                            x.0,
                                        )
                                    })
                                    .collect(),
                            },
                            None => {
                                panic!("Error: referencing undefined type: {}", name_token.slice)
                            }
                        }
                    }
                }
            },
            TypeSyntax::Array(base, _, length, _) => Self::Array(
                Box::new(Self::build(
                    symbol_scope,
                    sibling_scope_opaque_symbols,
                    *base,
                )),
                reduce_const_expr(&length).into_u32(),
            ),
            TypeSyntax::Ref {
                pointee_type,
                mut_token: None,
                decorator_token,
                ..
            } => Self::Ref(
                Box::new(Self::build(
                    symbol_scope,
                    sibling_scope_opaque_symbols,
                    *pointee_type,
                )),
                match decorator_token {
                    Some(t) if t.slice == "uniform" => RefStorageClass::Uniform,
                    Some(t) => todo!("Unknown StorageClass: {t:?}"),
                    // TODO: ここはコンテキストによる（シェーダエントリポイントならInputだし、そうでないならたいていLocal）
                    None => RefStorageClass::Input,
                },
            ),
            TypeSyntax::Ref {
                pointee_type,
                mut_token: Some(_),
                decorator_token,
                ..
            } => Self::MutableRef(
                Box::new(Self::build(
                    symbol_scope,
                    sibling_scope_opaque_symbols,
                    *pointee_type,
                )),
                match decorator_token {
                    Some(t) if t.slice == "uniform" => RefStorageClass::Uniform,
                    Some(t) => todo!("Unknown StorageClass: {t:?}"),
                    // TODO: ここはコンテキストによる（シェーダエントリポイントならInputだし、そうでないならたいていLocal）
                    None => RefStorageClass::Input,
                },
            ),
        }
    }

    pub fn instantiate(self, scope: &SymbolScope<'_, 's>) -> Self {
        match self {
            Self::UserDefined { name, generic_args } => {
                match scope.lookup_user_defined_type(name) {
                    Some((_, (_, UserDefinedType::Struct(members)))) => Self::Struct(
                        members
                            .iter()
                            .map(|x| UserDefinedStructMember {
                                attribute: x.attribute.clone(),
                                mutable: x.mutable,
                                name: x.name.clone(),
                                ty: x.ty.clone().instantiate(scope),
                            })
                            .collect(),
                    ),
                    None => Self::UserDefined { name, generic_args },
                }
            }
            Self::Function { args, output } => Self::Function {
                args: args.into_iter().map(|t| t.instantiate(scope)).collect(),
                output: output.map(|x| Box::new((*x).instantiate(scope))),
            },
            _ => self,
        }
    }

    pub const fn intrinsic_type(&self) -> Option<&IntrinsicType> {
        match self {
            Self::Intrinsic(it) => Some(it),
            _ => None,
        }
    }

    #[inline]
    pub fn try_cast_intrinsic_scalar(self, target: IntrinsicScalarType) -> Option<Self> {
        match self {
            Self::Intrinsic(it) => match it.try_cast_scalar(target) {
                Some(it) => Some(Self::Intrinsic(it)),
                None => None,
            },
            _ => None,
        }
    }

    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            Self::Intrinsic(x) => x.scalar_type(),
            _ => None,
        }
    }

    pub const fn intrinsic_composite_type_class(&self) -> Option<IntrinsicCompositeTypeClass> {
        match self {
            Self::Intrinsic(x) => x.composite_type_class(),
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            Self::Intrinsic(x) => x.vector_elements(),
            _ => None,
        }
    }

    pub const fn can_uniform_struct_member(&self) -> bool {
        match self {
            Self::Intrinsic(it) => it.can_uniform_struct_member(),
            Self::Struct(_) | Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            Self::Intrinsic(it) => it.std140_alignment(),
            Self::Struct(_) | Self::Tuple(_) => Some(16),
            _ => None,
        }
    }

    pub fn std140_size(&self) -> Option<usize> {
        match self {
            Self::Intrinsic(it) => it.std140_size(),
            Self::Struct(xs) => xs.iter().map(|x| x.ty.std140_size()).sum(),
            Self::Tuple(xs) => xs.iter().map(|x| x.std140_size()).sum(),
            _ => None,
        }
    }

    pub const fn is_scalar_type(&self) -> bool {
        match self {
            Self::Intrinsic(it) => it.is_scalar_type(),
            _ => false,
        }
    }

    pub const fn is_vector_type(&self) -> bool {
        match self {
            Self::Intrinsic(it) => it.is_vector_type(),
            _ => false,
        }
    }

    pub const fn is_matrix_type(&self) -> bool {
        match self {
            Self::Intrinsic(it) => it.is_matrix_type(),
            _ => false,
        }
    }

    pub fn make_spv_type(&self, scope: &SymbolScope<'_, 's>) -> spv::Type {
        match self {
            Self::Intrinsic(it) => it.make_spv_type(),
            &Self::Array(ref t, len) => spv::Type::Array {
                element_type: Box::new(t.make_spv_type(scope)),
                length: Box::new(spv::TypeArrayLength::ConstExpr(len.into())),
            },
            Self::Tuple(xs) => spv::Type::Struct {
                decorations: vec![],
                member_types: xs
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.std140_alignment().expect("cannot a member of a struct"),
                        );
                        *top = offs + x.std140_size().expect("cannot a member of a struct");
                        let ty = x.make_spv_type(scope);
                        let mut decorations = vec![spv::Decorate::Offset(offs as _)];
                        if let spv::Type::Matrix(ref c, _) = ty {
                            decorations.extend([
                                spv::Decorate::ColMajor,
                                spv::Decorate::MatrixStride(c.matrix_stride().unwrap()),
                            ]);
                        }

                        Some(spv::TypeStructMember { ty, decorations })
                    })
                    .collect(),
            },
            Self::Function { args, output } => spv::Type::Function {
                return_type: Box::new(
                    output
                        .as_ref()
                        .map_or(spv::Type::Void, |o| o.make_spv_type(scope)),
                ),
                parameter_types: args.iter().map(|x| x.make_spv_type(scope)).collect(),
            },
            Self::Struct(members) => spv::Type::Struct {
                decorations: vec![],
                member_types: members
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.ty.std140_alignment()
                                .expect("cannot a member of a struct"),
                        );
                        *top = offs + x.ty.std140_size().expect("cannot a member of a struct");
                        let ty = x.ty.make_spv_type(scope);
                        let mut decorations = vec![spv::Decorate::Offset(offs as _)];
                        if let spv::Type::Matrix(ref c, _) = ty {
                            decorations.extend([
                                spv::Decorate::ColMajor,
                                spv::Decorate::MatrixStride(c.matrix_stride().unwrap()),
                            ]);
                        }

                        Some(spv::TypeStructMember { ty, decorations })
                    })
                    .collect(),
            },
            Self::UserDefined { name, .. } => match scope.lookup_user_defined_type(name) {
                None => spv::Type::Opaque {
                    name: (*name).into(),
                },
                Some((_, (_, UserDefinedType::Struct(members)))) => spv::Type::Struct {
                    decorations: vec![],
                    member_types: members
                        .iter()
                        .scan(0, |top, x| {
                            let offs = roundup2(
                                *top,
                                x.ty.std140_alignment()
                                    .expect("cannot a member of a struct"),
                            );
                            *top = offs + x.ty.std140_size().expect("cannot a member of a struct");
                            let ty = x.ty.make_spv_type(scope);
                            let mut decorations = vec![spv::Decorate::Offset(offs as _)];
                            if let spv::Type::Matrix(ref c, _) = ty {
                                decorations.extend([
                                    spv::Decorate::ColMajor,
                                    spv::Decorate::MatrixStride(c.matrix_stride().unwrap()),
                                ]);
                            }

                            Some(spv::TypeStructMember { ty, decorations })
                        })
                        .collect(),
                },
            },
            Self::Ref(_, _) => {
                unreachable!("ref type")
            }
            Self::MutableRef(_, _) => unreachable!("mutable ref type"),
            Self::IntrinsicTypeConstructor(_) => {
                unreachable!("non-reduced intrinsic type construction")
            }
            Self::Never => unreachable!("type inference has error"),
            Self::Generic { .. } => unreachable!("uninstantiated generic type"),
            Self::GenericVar(_) => unreachable!("uninstantiated generic var"),
            Self::OverloadedFunctions(_) => unreachable!("unresolved overloads"),
        }
    }

    #[inline(always)]
    pub fn imm_ref(self, storage_class: RefStorageClass) -> Self {
        Self::Ref(Box::new(self), storage_class)
    }

    #[inline(always)]
    pub fn mutable_ref(self, storage_class: RefStorageClass) -> Self {
        Self::MutableRef(Box::new(self), storage_class)
    }

    #[inline(always)]
    pub fn dereference(self) -> Option<Self> {
        match self {
            Self::Ref(inner, _) => Some(*inner),
            Self::MutableRef(inner, _) => Some(*inner),
            _ => None,
        }
    }

    #[inline(always)]
    pub const fn as_dereferenced(&self) -> Option<&Self> {
        match self {
            Self::Ref(inner, _) | Self::MutableRef(inner, _) => Some(&**inner),
            _ => None,
        }
    }
}
impl From<IntrinsicType> for ConcreteType<'_> {
    #[inline(always)]
    fn from(value: IntrinsicType) -> Self {
        Self::Intrinsic(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserDefinedStructMember<'s> {
    pub attribute: SymbolAttribute,
    pub mutable: bool,
    pub name: SourceRefSliceEq<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub enum UserDefinedType<'s> {
    Struct(Vec<UserDefinedStructMember<'s>>),
}
