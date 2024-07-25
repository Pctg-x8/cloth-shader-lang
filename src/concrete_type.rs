use std::collections::HashSet;

use crate::{
    symbol::meta::SymbolAttribute, parser::TypeSyntax, scope::SymbolScope,
    source_ref::SourceRefSliceEq, spirv as spv, utils::roundup2,
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
    Sampler1D,
    Sampler2D,
    Sampler3D,
    Texture1D,
    Texture2D,
    Texture3D,
    SubpassInput,
}
impl IntrinsicType {
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
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
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
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
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
            Self::UInt3 | Self::SInt3 | Self::Float3 | Self::UInt4 | Self::SInt4 | Self::Float4 => {
                Some(16)
            }
            Self::Float2x2 | Self::Float2x3 | Self::Float2x4 => Some(16),
            Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => Some(16),
            // samplers/image refs cannot be a member of uniform struct
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => None,
        }
    }

    pub fn make_spv_type(&self) -> spv::Type {
        match self {
            Self::Unit => spv::Type::Void,
            Self::Bool => spv::Type::Bool,
            Self::UInt => spv::Type::uint(32),
            Self::SInt => spv::Type::sint(32),
            Self::Float => spv::Type::float(32),
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
            Self::Sampler1D => unreachable!("deprecated"),
            Self::Sampler2D => unreachable!("deprecated"),
            Self::Sampler3D => unreachable!("deprecated"),
            Self::Texture1D => unimplemented!(),
            Self::Texture2D => unimplemented!(),
            Self::Texture3D => unimplemented!(),
            Self::SubpassInput => spv::Type::subpass_data_image_type(),
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
    Function {
        args: Vec<ConcreteType<'s>>,
        output: Option<Box<ConcreteType<'s>>>,
    },
    IntrinsicTypeConstructor(IntrinsicType),
    Never,
}
impl<'s> ConcreteType<'s> {
    pub fn build(
        symbol_scope: &SymbolScope<'_, 's>,
        sibling_scope_opaque_symbols: &HashSet<&'s str>,
        t: TypeSyntax<'s>,
    ) -> Self {
        match t.name_token.slice {
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
            "Sampler1D" => Self::Intrinsic(IntrinsicType::Sampler1D),
            "Sampler2D" => Self::Intrinsic(IntrinsicType::Sampler2D),
            "Sampler3D" => Self::Intrinsic(IntrinsicType::Sampler3D),
            "Texture1D" => Self::Intrinsic(IntrinsicType::Texture1D),
            "Texture2D" => Self::Intrinsic(IntrinsicType::Texture2D),
            "Texture3D" => Self::Intrinsic(IntrinsicType::Texture3D),
            "SubpassInput" => Self::Intrinsic(IntrinsicType::SubpassInput),
            name => {
                if sibling_scope_opaque_symbols.contains(name) {
                    ConcreteType::UserDefined {
                        name: t.name_token.slice,
                        generic_args: t
                            .generic_args
                            .map_or_else(Vec::new, |x| x.args)
                            .into_iter()
                            .map(|x| {
                                ConcreteType::build(symbol_scope, sibling_scope_opaque_symbols, x.0)
                            })
                            .collect(),
                    }
                } else {
                    match symbol_scope.lookup_user_defined_type(name) {
                        Some(_) => ConcreteType::UserDefined {
                            name: t.name_token.slice,
                            generic_args: t
                                .generic_args
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
                        None => panic!("Error: referencing undefined type: {}", t.name_token.slice),
                    }
                }
            }
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
                                name: x.name.clone(),
                                ty: x.ty.clone().instantiate(scope),
                            })
                            .collect(),
                    ),
                    None => Self::UserDefined { name, generic_args },
                }
            }
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

    pub fn make_spv_type(&self, scope: &SymbolScope<'_, 's>) -> spv::Type {
        match self {
            Self::Intrinsic(it) => it.make_spv_type(),
            Self::Tuple(xs) => spv::Type::Struct {
                member_types: xs
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.std140_alignment().expect("cannot a member of a struct"),
                        );
                        *top = offs + x.std140_size().expect("cannot a member of a struct");

                        Some(spv::TypeStructMember {
                            ty: x.make_spv_type(scope),
                            decorations: vec![spv::Decorate::Offset(offs as _)],
                        })
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
                member_types: members
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.ty.std140_alignment()
                                .expect("cannot a member of a struct"),
                        );
                        *top = offs + x.ty.std140_size().expect("cannot a member of a struct");

                        Some(spv::TypeStructMember {
                            ty: x.ty.make_spv_type(scope),
                            decorations: vec![spv::Decorate::Offset(offs as _)],
                        })
                    })
                    .collect(),
            },
            Self::UserDefined { name, .. } => match scope.lookup_user_defined_type(name) {
                None => spv::Type::Opaque {
                    name: (*name).into(),
                },
                Some((_, (_, UserDefinedType::Struct(members)))) => spv::Type::Struct {
                    member_types: members
                        .iter()
                        .scan(0, |top, x| {
                            let offs = roundup2(
                                *top,
                                x.ty.std140_alignment()
                                    .expect("cannot a member of a struct"),
                            );
                            *top = offs + x.ty.std140_size().expect("cannot a member of a struct");

                            Some(spv::TypeStructMember {
                                ty: x.ty.make_spv_type(scope),
                                decorations: vec![spv::Decorate::Offset(offs as _)],
                            })
                        })
                        .collect(),
                },
            },
            Self::IntrinsicTypeConstructor(_) => {
                unreachable!("non-reduced intrinsic type construction")
            }
            Self::Never => unreachable!("type inference has error"),
            Self::Generic { .. } => unreachable!("uninstantiated generic type"),
            Self::GenericVar(_) => unreachable!("uninstantiated generic var"),
            Self::UnknownIntClass => unreachable!("left UnknownIntClass"),
            Self::UnknownNumberClass => unreachable!("left UnknownNumberClass"),
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

pub enum BinaryOpTypeConversion {
    NoConversion,
    CastLeftHand,
    CastRightHand,
    CastBoth,
    InstantiateAndCastLeftHand(IntrinsicType),
    InstantiateAndCastRightHand(IntrinsicType),
    InstantiateRightAndCastLeftHand(IntrinsicType),
    InstantiateLeftAndCastRightHand(IntrinsicType),
    InstantiateLeftHand(IntrinsicType),
    InstantiateRightHand(IntrinsicType),
}
impl<'s> ConcreteType<'s> {
    pub fn arithmetic_compare_op_type_conversion(
        self,
        rhs: Self,
    ) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    // instantiate left
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::SInt),
                        b,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::UInt),
                        b,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::SInt),
                        b,
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
                            b,
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
                        a,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::SInt),
                        a,
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
                            a,
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

    pub fn bitwise_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
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
                    (IntrinsicScalarType::Bool, _) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (_, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserDefinedStructMember<'s> {
    pub attribute: SymbolAttribute,
    pub name: SourceRefSliceEq<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub enum UserDefinedType<'s> {
    Struct(Vec<UserDefinedStructMember<'s>>),
}
