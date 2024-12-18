use std::collections::HashSet;

use crate::{
    const_expr::reduce_const_expr, parser::TypeSyntax, scope::SymbolScope,
    source_ref::SourceRefSliceEq, spirv as spv, symbol::meta::SymbolAttribute, utils::roundup2,
};

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
    Ref(Box<ConcreteType<'s>>),
    MutableRef(Box<ConcreteType<'s>>),
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
            // TODO: decoratorでポインティングデータの所属クラスを引いてくる
            TypeSyntax::Ref {
                pointee_type,
                mut_token: None,
                ..
            } => Self::Ref(Box::new(Self::build(
                symbol_scope,
                sibling_scope_opaque_symbols,
                *pointee_type,
            ))),
            TypeSyntax::Ref {
                pointee_type,
                mut_token: Some(_),
                ..
            } => Self::MutableRef(Box::new(Self::build(
                symbol_scope,
                sibling_scope_opaque_symbols,
                *pointee_type,
            ))),
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
            Self::Ref(_) => {
                unreachable!("ref type")
            }
            Self::MutableRef(_) => unreachable!("mutable ref type"),
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
    pub fn imm_ref(self) -> Self {
        Self::Ref(Box::new(self))
    }

    #[inline(always)]
    pub fn mutable_ref(self) -> Self {
        Self::MutableRef(Box::new(self))
    }

    #[inline(always)]
    pub fn dereference(self) -> Option<Self> {
        match self {
            Self::Ref(inner) => Some(*inner),
            Self::MutableRef(inner) => Some(*inner),
            _ => None,
        }
    }

    #[inline(always)]
    pub const fn as_dereferenced(&self) -> Option<&Self> {
        match self {
            Self::Ref(inner) | Self::MutableRef(inner) => Some(&**inner),
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

/// 演算子の項の値変換指示
pub enum BinaryOpScalarConversion {
    /// 変換なし
    None,
    /// UnknownNumberClassに昇格（主にUnknownIntNumberに対して）
    PromoteUnknownNumber,
    /// 指定した型にキャスト
    Cast(IntrinsicScalarType),
    /// 指定した型にUnknownHogehogeClassの値を実体化する
    Instantiate(IntrinsicScalarType),
}

pub enum BinaryOpTermConversion {
    NoConversion,
    PromoteUnknownNumber,
    Cast(IntrinsicType),
    Instantiate(IntrinsicType),
    Distribute(IntrinsicType),
    CastAndDistribute(IntrinsicType, u32),
    InstantiateAndDistribute(IntrinsicType, u32),
}
impl BinaryOpTermConversion {
    const fn distribute(self, to: IntrinsicType, component_count: u32) -> Self {
        match self {
            Self::NoConversion => Self::Distribute(to),
            Self::Cast(x) => Self::CastAndDistribute(x, component_count),
            Self::Instantiate(x) => Self::InstantiateAndDistribute(x, component_count),
            Self::Distribute(_) => Self::Distribute(to),
            Self::CastAndDistribute(x, _) => Self::CastAndDistribute(x, component_count),
            Self::InstantiateAndDistribute(x, _) => {
                Self::InstantiateAndDistribute(x, component_count)
            }
            Self::PromoteUnknownNumber => panic!("unknown op"),
        }
    }
}

/// 分配オペレーションの必要性を表す
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpValueDistributionRequirements {
    /// 左の項の分配が必要
    LeftTerm,
    /// 右の項の分配が必要
    RightTerm,
}

/// 二項演算の型変換/分配指示データ
pub struct BinaryOpTypeConversion2<'s> {
    /// 演算子の左の項の変換指示
    pub left_op: BinaryOpScalarConversion,
    /// 演算子の右の項の変換指示
    pub right_op: BinaryOpScalarConversion,
    /// どちらの項の値の分配すべきか Noneの場合は分配処理はなし
    pub dist: Option<BinaryOpValueDistributionRequirements>,
    /// この演算の最終的な型
    pub result_type: ConcreteType<'s>,
}

/// （乗算以外の）組み込み算術演算の自動型変換ルールの定義
///
/// ベクトル/行列との乗算に特殊なルールが存在するので、乗算は別で定義
pub fn arithmetic_compare_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ型

        if (lhs.scalar_type(), rhs.scalar_type())
            == (
                Some(IntrinsicScalarType::Bool),
                Some(IntrinsicScalarType::Bool),
            )
        {
            // Bool型どうしの算術演算はSIntに上げる
            return Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
                right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
                dist: None,
                result_type: ConcreteType::Intrinsic(
                    lhs.intrinsic_composite_type_class()
                        .expect("no intrinsic type?")
                        .combine_scalar(IntrinsicScalarType::SInt),
                ),
            });
        }

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()?,
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // 片方でもUnitの場合は変換なし（他のところで形マッチエラーにさせる）
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // boolの算術演算はSIntとして行う（型一致時の特殊ルール）
        (IntrinsicScalarType::Bool, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        // 同じ型だったら変換なし（この場合はDistributionだけが走る）
        (IntrinsicScalarType::UInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // UnknownHogehogeClassは適当にInstantiateさせて、あとはBoolとの演算ルールに従う
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        // 左の型に合わせるパターン
        (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 左の型に合わせるパターン（Instantiate）
        (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
        // 左の型に合わせるパターン（昇格）
        (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::PromoteUnknownNumber,
            l,
        ),
        // 右の型に合わせるパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右の型に合わせるパターン（Instantiate）
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右の型に合わせるパターン（昇格）
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::PromoteUnknownNumber,
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右をInstantiateしたうえで、すべての演算をFloatで行うパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            IntrinsicScalarType::Float,
        ),
        // 左をInstantiateしたうえで、すべての演算をFloatで行うパターン
        (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
            IntrinsicScalarType::Float,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

/// 組み込みビット演算の自動型変換ルールの定義
pub fn bitwise_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ型

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()
                .expect("no intrinsic type"),
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // 片方でもUnitの場合は変換なし（他のところで形マッチエラーにさせる）
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // 片方でもFloatの場合は演算不可 UnknownNumberClassも同様
        (IntrinsicScalarType::Float, _)
        | (_, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownNumberClass, _)
        | (_, IntrinsicScalarType::UnknownNumberClass) => return None,
        // 同じ型だったら変換なし（この場合はDistributionだけが走る）
        (IntrinsicScalarType::Bool, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // UnknownHogehogeClassは適当にInstantiateさせて、あとはBoolとの演算ルールに従う
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
            IntrinsicScalarType::UInt,
        ),
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
            IntrinsicScalarType::UInt,
        ),
        // 右に合わせるパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 左に合わせるパターン
        (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 左をInstantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右をInstantiateするパターン
        (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

pub fn logical_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    todo!("練り直し");
    /*match (lhs, rhs) {
        // between same type
        (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
        // between same length vectors
        (a, b) if a.vector_elements()? == b.vector_elements()? => {
            match (a.scalar_type()?, b.scalar_type()?) {
                // instantiate and cast
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                    BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::UInt),
                    a,
                )),
                (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                    BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::UInt),
                    a,
                )),
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass) => Some((
                    BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::Float),
                    a,
                )),
                (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool) => Some((
                    BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::Float),
                    a,
                )),
                // simple casting
                (IntrinsicScalarType::Bool, _) => Some((
                    BinaryOpTypeConversion::CastRightHand(IntrinsicType::Bool),
                    a,
                )),
                (_, IntrinsicScalarType::Bool) => {
                    Some((BinaryOpTypeConversion::CastLeftHand(IntrinsicType::Bool), b))
                }
                // never
                _ => None,
            }
        }
        // never
        _ => None,
    }*/
}

pub fn pow_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    todo!("練り直し");
    /*let (left_conversion, resulting_left_ty) = match self {
        Self::Intrinsic(IntrinsicType::Bool) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::UInt) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::SInt) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::Float) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float)
        }
        Self::Intrinsic(IntrinsicType::UInt2) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float2),
            IntrinsicType::Float2,
        ),
        Self::Intrinsic(IntrinsicType::SInt2) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float2),
            IntrinsicType::Float2,
        ),
        Self::Intrinsic(IntrinsicType::Float2) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float2)
        }
        Self::Intrinsic(IntrinsicType::UInt3) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float3),
            IntrinsicType::Float3,
        ),
        Self::Intrinsic(IntrinsicType::SInt3) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float3),
            IntrinsicType::Float3,
        ),
        Self::Intrinsic(IntrinsicType::Float3) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float3)
        }
        Self::Intrinsic(IntrinsicType::UInt4) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float4),
            IntrinsicType::Float4,
        ),
        Self::Intrinsic(IntrinsicType::SInt4) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float4),
            IntrinsicType::Float4,
        ),
        Self::Intrinsic(IntrinsicType::Float4) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float4)
        }
        Self::UnknownIntClass | Self::UnknownNumberClass => (
            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        _ => return None,
    };
    let (right_conversion, resulting_right_ty) = match rhs {
        Self::Intrinsic(IntrinsicType::Bool) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::UInt) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::SInt) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        Self::Intrinsic(IntrinsicType::Float) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float)
        }
        Self::Intrinsic(IntrinsicType::UInt2) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float2),
            IntrinsicType::Float2,
        ),
        Self::Intrinsic(IntrinsicType::SInt2) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float2),
            IntrinsicType::Float2,
        ),
        Self::Intrinsic(IntrinsicType::Float2) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float2)
        }
        Self::Intrinsic(IntrinsicType::UInt3) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float3),
            IntrinsicType::Float3,
        ),
        Self::Intrinsic(IntrinsicType::SInt3) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float3),
            IntrinsicType::Float3,
        ),
        Self::Intrinsic(IntrinsicType::Float3) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float3)
        }
        Self::Intrinsic(IntrinsicType::UInt4) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float4),
            IntrinsicType::Float4,
        ),
        Self::Intrinsic(IntrinsicType::SInt4) => (
            BinaryOpTermConversion::Cast(IntrinsicType::Float4),
            IntrinsicType::Float4,
        ),
        Self::Intrinsic(IntrinsicType::Float4) => {
            (BinaryOpTermConversion::NoConversion, IntrinsicType::Float4)
        }
        Self::UnknownIntClass | Self::UnknownNumberClass => (
            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
            IntrinsicType::Float,
        ),
        _ => return None,
    };

    match (resulting_left_ty, resulting_right_ty) {
        (a, b) if a == b => Some((left_conversion, right_conversion, resulting_left_ty.into())),
        (a, b) if a.is_scalar_type() && b.is_vector_type() => Some((
            left_conversion.distribute(b, b.vector_elements().unwrap() as _),
            right_conversion,
            resulting_right_ty.into(),
        )),
        (a, b) if a.is_vector_type() && b.is_scalar_type() => Some((
            left_conversion,
            right_conversion.distribute(a, a.vector_elements().unwrap() as _),
            resulting_left_ty.into(),
        )),
        _ => None,
    }*/
}

/// 乗算の自動型変換ルールの定義
pub fn multiply_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    match (lhs, rhs) {
        // between same type
        (a, b) if a == b => Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        }),
        // vector times scalar
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(a, _))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(b)),
        ) if a == b => {
            // both same element type and scalar type
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: BinaryOpScalarConversion::None,
                dist: None,
                result_type: lhs.clone(),
            })
        }
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(
                IntrinsicScalarType::Float,
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(r)),
        ) => {
            // force casting/instantiating right hand to float
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: if r.is_unknown_type() {
                    BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
                } else {
                    BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
                },
                dist: None,
                result_type: lhs.clone(),
            })
        }
        // matrix times scalar
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(a, _),
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(b)),
        ) if a == b => {
            // both same element type and scalar type
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: BinaryOpScalarConversion::None,
                dist: None,
                result_type: lhs.clone(),
            })
        }
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(IntrinsicScalarType::Float, _),
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(r)),
        ) => {
            // force casting/instantiating right hand to float
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: if r.is_unknown_type() {
                    BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
                } else {
                    BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
                },
                dist: None,
                result_type: lhs.clone(),
            })
        }
        // vector times matrix
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(rl)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(rr, c))),
        ) if rl == rr => Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: IntrinsicType::Vector(IntrinsicVectorType(rl.0, *c)).into(),
        }),
        // matrix times vector
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(IntrinsicScalarType::Float, r),
                cl,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(
                IntrinsicScalarType::Float,
                rr,
            ))),
        ) if cl == rr => Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: IntrinsicType::vec(*r).into(),
        }),
        // between same length vectors
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(vl)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(vr)),
        ) if vl.1 == vr.1 => {
            match (vl.0, vr.0) {
                // simple casting
                // empowered right hand
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::Bool, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::Float) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(vr.0),
                        right_op: BinaryOpScalarConversion::None,
                        dist: None,
                        result_type: rhs.clone(),
                    })
                }
                // empowered left hand
                (IntrinsicScalarType::Float, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::None,
                        right_op: BinaryOpScalarConversion::Cast(vl.0),
                        dist: None,
                        result_type: lhs.clone(),
                    })
                }
                // instantiate lhs, cast rhs, operate on uint type
                (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
                        right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
                        dist: None,
                        result_type: IntrinsicType::Vector(vl.with_type(IntrinsicScalarType::UInt))
                            .into(),
                    })
                }
                // instantiate rhs, cast lhs, operate on uint type
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
                        right_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
                        dist: None,
                        result_type: IntrinsicType::Vector(vl.with_type(IntrinsicScalarType::UInt))
                            .into(),
                    })
                }
                // simply instantiate to rhs type
                (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(vr.0),
                        right_op: BinaryOpScalarConversion::None,
                        dist: None,
                        result_type: rhs.clone(),
                    })
                }
                // simply instantiate to lhs type
                (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::None,
                        right_op: BinaryOpScalarConversion::Instantiate(vl.0),
                        dist: None,
                        result_type: lhs.clone(),
                    })
                }
                // instantiate lhs to float, cast rhs to float
                (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
                        right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
                        dist: None,
                        result_type: IntrinsicType::Vector(
                            vl.with_type(IntrinsicScalarType::Float),
                        )
                        .into(),
                    })
                }
                // instantiate rhs to float, cast lhs to float
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
                        right_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
                        dist: None,
                        result_type: IntrinsicType::Vector(
                            vl.with_type(IntrinsicScalarType::Float),
                        )
                        .into(),
                    })
                }
                // never
                _ => None,
            }
        }
        // never
        _ => None,
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
