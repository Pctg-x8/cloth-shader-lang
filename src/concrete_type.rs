use std::collections::HashSet;

use crate::{
    const_expr::reduce_const_expr, parser::TypeSyntax, scope::SymbolScope,
    source_ref::SourceRefSliceEq, spirv as spv, symbol::meta::SymbolAttribute, utils::roundup2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicScalarType {
    Unit,
    Bool,
    UInt,
    SInt,
    Float,
    UnknownIntClass,
    UnknownNumberClass,
}
impl IntrinsicScalarType {
    #[inline(always)]
    pub const fn of_vector(self, count: u8) -> Option<IntrinsicType> {
        match (self, count) {
            (Self::Unit, 0) => Some(IntrinsicType::Unit),
            (Self::Bool, 1) => Some(IntrinsicType::Bool),
            (Self::UInt, 1) => Some(IntrinsicType::UInt),
            (Self::SInt, 1) => Some(IntrinsicType::SInt),
            (Self::Float, 1) => Some(IntrinsicType::Float),
            (Self::UInt, 2) => Some(IntrinsicType::UInt2),
            (Self::SInt, 2) => Some(IntrinsicType::SInt2),
            (Self::Float, 2) => Some(IntrinsicType::Float2),
            (Self::UInt, 3) => Some(IntrinsicType::UInt3),
            (Self::SInt, 3) => Some(IntrinsicType::SInt3),
            (Self::Float, 3) => Some(IntrinsicType::Float3),
            (Self::UInt, 4) => Some(IntrinsicType::UInt4),
            (Self::SInt, 4) => Some(IntrinsicType::SInt4),
            (Self::Float, 4) => Some(IntrinsicType::Float4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicType {
    Unit,
    Bool,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    SInt,
    SInt2,
    SInt3,
    SInt4,
    Float,
    Float2,
    Float3,
    Float4,
    Float2x2,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x3,
    Float3x4,
    Float4x2,
    Float4x3,
    Float4x4,
    Texture1D,
    Texture2D,
    Texture3D,
    Image1D,
    Image2D,
    Image3D,
    SubpassInput,
}
impl IntrinsicType {
    pub const fn of_vector(self, component_count: u8) -> Option<Self> {
        match (self, component_count) {
            (_, 1) => Some(self),
            (Self::SInt, 2) => Some(Self::SInt2),
            (Self::SInt, 3) => Some(Self::SInt3),
            (Self::SInt, 4) => Some(Self::SInt4),
            (Self::UInt, 2) => Some(Self::UInt2),
            (Self::UInt, 3) => Some(Self::UInt3),
            (Self::UInt, 4) => Some(Self::UInt4),
            (Self::Float, 2) => Some(Self::Float2),
            (Self::Float, 3) => Some(Self::Float3),
            (Self::Float, 4) => Some(Self::Float4),
            _ => None,
        }
    }

    pub const fn bvec(component_count: u8) -> Option<Self> {
        Self::Bool.of_vector(component_count)
    }

    pub const fn uvec(component_count: u8) -> Option<Self> {
        Self::UInt.of_vector(component_count)
    }

    pub const fn ivec(component_count: u8) -> Option<Self> {
        Self::SInt.of_vector(component_count)
    }

    pub const fn vec(component_count: u8) -> Option<Self> {
        Self::Float.of_vector(component_count)
    }

    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            Self::Unit => Some(IntrinsicScalarType::Unit),
            Self::Bool => Some(IntrinsicScalarType::Bool),
            Self::UInt | Self::UInt2 | Self::UInt3 | Self::UInt4 => Some(IntrinsicScalarType::UInt),
            Self::SInt | Self::SInt2 | Self::SInt3 | Self::SInt4 => Some(IntrinsicScalarType::SInt),
            Self::Float | Self::Float2 | Self::Float3 | Self::Float4 => {
                Some(IntrinsicScalarType::Float)
            }
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            Self::Unit => Some(0),
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(1),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(2),
            Self::UInt3 | Self::SInt3 | Self::Float3 => Some(3),
            Self::UInt4 | Self::SInt4 | Self::Float4 => Some(4),
            _ => None,
        }
    }

    pub const fn can_uniform_struct_member(&self) -> bool {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => false,
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
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(8),
            Self::UInt3 | Self::SInt3 | Self::Float3 | Self::UInt4 | Self::SInt4 | Self::Float4 => {
                Some(16)
            }
            Self::Float2x2 | Self::Float2x3 | Self::Float2x4 => Some(8),
            Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => Some(16),
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
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(8),
            Self::UInt3 | Self::SInt3 | Self::Float3 => Some(12),
            Self::UInt4 | Self::SInt4 | Self::Float4 => Some(16),
            Self::Float2x2 | Self::Float2x3 | Self::Float2x4 => Some(16),
            Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => Some(16),
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
        match self {
            Self::Bool | Self::UInt | Self::SInt | Self::Float => true,
            _ => false,
        }
    }

    pub const fn is_vector_type(&self) -> bool {
        match self {
            Self::Float2 | Self::Float3 | Self::Float4 => true,
            _ => false,
        }
    }

    pub const fn is_matrix_type(&self) -> bool {
        match self {
            Self::Float2x2
            | Self::Float2x3
            | Self::Float2x4
            | Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => true,
            _ => false,
        }
    }

    pub fn make_spv_type(&self) -> spv::Type {
        match self {
            Self::Unit => spv::Type::Void,
            Self::Bool => spv::ScalarType::Bool.into(),
            Self::UInt => spv::ScalarType::Int(32, false).into(),
            Self::SInt => spv::ScalarType::Int(32, true).into(),
            Self::Float => spv::ScalarType::Float(32).into(),
            Self::UInt2 => Self::UInt.make_spv_type().of_vector(2),
            Self::UInt3 => Self::UInt.make_spv_type().of_vector(3),
            Self::UInt4 => Self::UInt.make_spv_type().of_vector(4),
            Self::SInt2 => Self::SInt.make_spv_type().of_vector(2),
            Self::SInt3 => Self::SInt.make_spv_type().of_vector(3),
            Self::SInt4 => Self::SInt.make_spv_type().of_vector(4),
            Self::Float2 => Self::Float.make_spv_type().of_vector(2),
            Self::Float3 => Self::Float.make_spv_type().of_vector(3),
            Self::Float4 => Self::Float.make_spv_type().of_vector(4),
            Self::Float2x2 => Self::Float.make_spv_type().of_matrix(2, 2),
            Self::Float2x3 => Self::Float.make_spv_type().of_matrix(2, 3),
            Self::Float2x4 => Self::Float.make_spv_type().of_matrix(2, 4),
            Self::Float3x2 => Self::Float.make_spv_type().of_matrix(3, 2),
            Self::Float3x3 => Self::Float.make_spv_type().of_matrix(3, 3),
            Self::Float3x4 => Self::Float.make_spv_type().of_matrix(3, 4),
            Self::Float4x2 => Self::Float.make_spv_type().of_matrix(4, 2),
            Self::Float4x3 => Self::Float.make_spv_type().of_matrix(4, 3),
            Self::Float4x4 => Self::Float.make_spv_type().of_matrix(4, 4),
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
    UnknownIntClass,
    UnknownNumberClass,
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
                "UInt" => Self::Intrinsic(IntrinsicType::UInt),
                "UInt2" => Self::Intrinsic(IntrinsicType::UInt2),
                "UInt3" => Self::Intrinsic(IntrinsicType::UInt3),
                "UInt4" => Self::Intrinsic(IntrinsicType::UInt4),
                "SInt" | "Int" => Self::Intrinsic(IntrinsicType::SInt),
                "SInt2" | "Int2" => Self::Intrinsic(IntrinsicType::SInt2),
                "SInt3" | "Int3" => Self::Intrinsic(IntrinsicType::SInt3),
                "SInt4" | "Int4" => Self::Intrinsic(IntrinsicType::SInt4),
                "Float" => Self::Intrinsic(IntrinsicType::Float),
                "Float2" => Self::Intrinsic(IntrinsicType::Float2),
                "Float3" => Self::Intrinsic(IntrinsicType::Float3),
                "Float4" => Self::Intrinsic(IntrinsicType::Float4),
                "Float2x2" => Self::Intrinsic(IntrinsicType::Float2x2),
                "Float2x3" => Self::Intrinsic(IntrinsicType::Float2x3),
                "Float2x4" => Self::Intrinsic(IntrinsicType::Float2x4),
                "Float3x2" => Self::Intrinsic(IntrinsicType::Float3x2),
                "Float3x3" => Self::Intrinsic(IntrinsicType::Float3x3),
                "Float3x4" => Self::Intrinsic(IntrinsicType::Float3x4),
                "Float4x2" => Self::Intrinsic(IntrinsicType::Float4x2),
                "Float4x3" => Self::Intrinsic(IntrinsicType::Float4x3),
                "Float4x4" => Self::Intrinsic(IntrinsicType::Float4x4),
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

    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            Self::Intrinsic(x) => x.scalar_type(),
            Self::UnknownIntClass => Some(IntrinsicScalarType::UnknownIntClass),
            Self::UnknownNumberClass => Some(IntrinsicScalarType::UnknownNumberClass),
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            Self::Intrinsic(x) => x.vector_elements(),
            Self::UnknownIntClass | Self::UnknownNumberClass => Some(1),
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
            Self::UnknownIntClass | Self::UnknownNumberClass => true,
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

    pub const fn is_unknown_type_class(&self) -> bool {
        matches!(self, Self::UnknownIntClass | Self::UnknownNumberClass)
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
            Self::UnknownIntClass => unreachable!("left UnknownIntClass"),
            Self::UnknownNumberClass => unreachable!("left UnknownNumberClass"),
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
impl From<IntrinsicScalarType> for ConcreteType<'_> {
    #[inline(always)]
    fn from(value: IntrinsicScalarType) -> Self {
        match value {
            IntrinsicScalarType::Unit => Self::Intrinsic(IntrinsicType::Unit),
            IntrinsicScalarType::Bool => Self::Intrinsic(IntrinsicType::Bool),
            IntrinsicScalarType::UInt => Self::Intrinsic(IntrinsicType::UInt),
            IntrinsicScalarType::SInt => Self::Intrinsic(IntrinsicType::SInt),
            IntrinsicScalarType::Float => Self::Intrinsic(IntrinsicType::Float),
            IntrinsicScalarType::UnknownIntClass => Self::UnknownIntClass,
            IntrinsicScalarType::UnknownNumberClass => Self::UnknownNumberClass,
        }
    }
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

pub enum BinaryOpTypeConversion {
    NoConversion,
    CastLeftHand(IntrinsicType),
    CastRightHand(IntrinsicType),
    InstantiateAndCastLeftHand(IntrinsicType),
    InstantiateAndCastRightHand(IntrinsicType),
    InstantiateRightAndCastLeftHand(IntrinsicType),
    InstantiateLeftAndCastRightHand(IntrinsicType),
    InstantiateLeftHand(IntrinsicType),
    InstantiateRightHand(IntrinsicType),
}
impl<'s> ConcreteType<'s> {
    pub fn multiply_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (&self, &rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, self)),
            // vector times scalar
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float2), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (ConcreteType::Intrinsic(IntrinsicType::Float2), ConcreteType::UnknownNumberClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float3), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (ConcreteType::Intrinsic(IntrinsicType::Float3), ConcreteType::UnknownNumberClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float4), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (ConcreteType::Intrinsic(IntrinsicType::Float4), ConcreteType::UnknownNumberClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            // matrix times scalar
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float2x2), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float3x2), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float4x2), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float2x3), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float3x3), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float4x3), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float2x4), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float3x4), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::Intrinsic(IntrinsicType::Bool),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::Intrinsic(IntrinsicType::UInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::Intrinsic(IntrinsicType::SInt),
            ) => Some((
                BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                self,
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::Intrinsic(IntrinsicType::Float),
            ) => Some((BinaryOpTypeConversion::NoConversion, self)),
            (ConcreteType::Intrinsic(IntrinsicType::Float4x4), ConcreteType::UnknownIntClass) => {
                Some((
                    BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                    self,
                ))
            }
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::UnknownNumberClass,
            ) => Some((
                BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                self,
            )),
            // vector times matrix
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2),
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3),
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4),
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            // matrix times vector
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x2),
                ConcreteType::Intrinsic(IntrinsicType::Float2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x2),
                ConcreteType::Intrinsic(IntrinsicType::Float2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x2),
                ConcreteType::Intrinsic(IntrinsicType::Float2),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x3),
                ConcreteType::Intrinsic(IntrinsicType::Float3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x3),
                ConcreteType::Intrinsic(IntrinsicType::Float3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x3),
                ConcreteType::Intrinsic(IntrinsicType::Float3),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float2x4),
                ConcreteType::Intrinsic(IntrinsicType::Float4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float2.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float3x4),
                ConcreteType::Intrinsic(IntrinsicType::Float4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float3.into(),
            )),
            (
                ConcreteType::Intrinsic(IntrinsicType::Float4x4),
                ConcreteType::Intrinsic(IntrinsicType::Float4),
            ) => Some((
                BinaryOpTypeConversion::NoConversion,
                IntrinsicType::Float4.into(),
            )),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::Float),
                        rhs,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::Float),
                        rhs,
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::Float),
                        rhs,
                    )),
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::UInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::SInt),
                        self,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                        self,
                    )),
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::SInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::CastLeftHand(IntrinsicType::SInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                        self,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::UInt),
                        self,
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::SInt),
                        self,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::Float),
                        self,
                    )),
                    // instantiate left
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::SInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::UInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::SInt),
                        rhs,
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => {
                        Some((
                            BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::Float),
                            rhs,
                        ))
                    }
                    // instantiate right
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::SInt),
                        self,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::UInt),
                        self,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::SInt),
                        self,
                    )),
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => {
                        Some((
                            BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                            self,
                        ))
                    }
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }

    pub fn arithmetic_compare_op_type_conversion(
        self,
        rhs: Self,
    ) -> Option<(BinaryOpTermConversion, BinaryOpTermConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((
                BinaryOpTermConversion::NoConversion,
                BinaryOpTermConversion::NoConversion,
                a,
            )),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                let element_count = a.vector_elements()?;

                match a.scalar_type()? {
                    IntrinsicScalarType::Unit => None,
                    IntrinsicScalarType::Bool => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::uvec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::Instantiate(IntrinsicType::SInt),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::SInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::NoConversion,
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Instantiate(IntrinsicType::SInt),
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::UInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Cast(IntrinsicType::uvec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::NoConversion,
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Instantiate(IntrinsicType::UInt),
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::Float => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool
                        | IntrinsicScalarType::SInt
                        | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::NoConversion,
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            a.clone(),
                        )),
                    },
                    IntrinsicScalarType::UnknownIntClass => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::SInt),
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::SInt),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::UInt),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::NoConversion,
                            ConcreteType::UnknownIntClass,
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::PromoteUnknownNumber,
                            BinaryOpTermConversion::NoConversion,
                            ConcreteType::UnknownNumberClass,
                        )),
                    },
                    IntrinsicScalarType::UnknownNumberClass => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool
                        | IntrinsicScalarType::SInt
                        | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Instantiate(IntrinsicType::Float),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::PromoteUnknownNumber,
                            ConcreteType::UnknownNumberClass,
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::NoConversion,
                            ConcreteType::UnknownNumberClass,
                        )),
                    },
                }
            }
            // extend one side
            (a, b) if a.is_vector_type() && b.is_scalar_type() => {
                let element_count = a.vector_elements().unwrap();

                match a.scalar_type()? {
                    IntrinsicScalarType::Unit => None,
                    IntrinsicScalarType::Bool => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::ivec(element_count)?,
                                element_count as _,
                            ),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::uvec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::uvec(element_count)?),
                            IntrinsicType::uvec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::SInt,
                                element_count as _,
                            ),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::SInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::ivec(element_count)?,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Distribute(IntrinsicType::ivec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::SInt,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::UInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::uvec(element_count)?,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Distribute(IntrinsicType::uvec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::UnknownIntClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::UInt,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                    },
                    IntrinsicScalarType::Float => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool
                        | IntrinsicScalarType::SInt
                        | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::vec(element_count)?,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            a.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => Some((
                            BinaryOpTermConversion::NoConversion,
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            a.clone(),
                        )),
                    },
                    IntrinsicScalarType::UnknownIntClass
                    | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                }
            }
            (a, b) if a.is_scalar_type() && b.is_vector_type() => {
                let element_count = b.vector_elements().unwrap();

                match a.scalar_type()? {
                    IntrinsicScalarType::Unit => None,
                    IntrinsicScalarType::Bool => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::ivec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::ivec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::uvec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::vec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                    IntrinsicScalarType::SInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::ivec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::vec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                    IntrinsicScalarType::UInt => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::uvec(element_count)?),
                            BinaryOpTermConversion::Cast(IntrinsicType::uvec(element_count)?),
                            IntrinsicType::uvec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::ivec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::uvec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::CastAndDistribute(
                                IntrinsicType::vec(element_count)?,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                    IntrinsicScalarType::Float => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool
                        | IntrinsicScalarType::SInt
                        | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            IntrinsicType::vec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::Distribute(IntrinsicType::vec(element_count)?),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                    IntrinsicScalarType::UnknownIntClass => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::SInt,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::Cast(IntrinsicType::ivec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::SInt => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::SInt,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::UInt,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                    IntrinsicScalarType::UnknownNumberClass => match b.scalar_type()? {
                        IntrinsicScalarType::Unit => None,
                        IntrinsicScalarType::Bool
                        | IntrinsicScalarType::SInt
                        | IntrinsicScalarType::UInt => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::Cast(IntrinsicType::vec(element_count)?),
                            IntrinsicType::ivec(element_count)?.into(),
                        )),
                        IntrinsicScalarType::Float => Some((
                            BinaryOpTermConversion::InstantiateAndDistribute(
                                IntrinsicType::Float,
                                element_count as _,
                            ),
                            BinaryOpTermConversion::NoConversion,
                            b.clone(),
                        )),
                        IntrinsicScalarType::UnknownIntClass
                        | IntrinsicScalarType::UnknownNumberClass => unreachable!(),
                    },
                }
            }
            // never
            _ => None,
        }
    }

    pub fn bitwise_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand(IntrinsicType::UInt), b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand(IntrinsicType::UInt), b))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand(IntrinsicType::SInt), b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::CastRightHand(IntrinsicType::SInt),
                        a,
                    )),
                    // instantiate left
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::SInt),
                        IntrinsicType::SInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::UInt),
                        b,
                    )),
                    // instantiate right
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::SInt),
                        IntrinsicType::SInt.into(),
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }

    pub fn logical_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
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
        }
    }

    pub fn pow_op_type_conversion(
        self,
        rhs: Self,
    ) -> Option<(BinaryOpTermConversion, BinaryOpTermConversion, Self)> {
        let (left_conversion, resulting_left_ty) = match self {
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
        }
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
