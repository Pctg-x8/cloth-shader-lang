#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum TypeArrayLength {
    ConstExpr(super::Constant),
    SpecConstantID(u32),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct TypeStructMember {
    pub ty: Type,
    pub decorations: Vec<super::Decorate>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ScalarType {
    Bool,
    /// (width, signedness)
    Int(u32, bool),
    Float(u32),
}
impl ScalarType {
    pub const fn of_vector(self, count: VectorSize) -> VectorType {
        VectorType(self, count)
    }

    pub const fn of_matrix(
        self,
        row_count: VectorSize,
        column_count: MatrixColumnCount,
    ) -> MatrixType {
        self.of_vector(row_count).of_matrix(column_count)
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum VectorSize {
    Two = 2,
    Three = 3,
    Four = 4,
}
impl VectorSize {
    #[inline(always)]
    pub const fn count(self) -> u8 {
        self as _
    }
}
impl TryFrom<u8> for VectorSize {
    type Error = u8;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            _ => Err(value),
        }
    }
}
impl TryFrom<u32> for VectorSize {
    type Error = u32;

    #[inline]
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            _ => Err(value),
        }
    }
}
impl From<VectorSize> for u32 {
    #[inline(always)]
    fn from(value: VectorSize) -> Self {
        value as _
    }
}
impl From<MatrixColumnCount> for VectorSize {
    #[inline(always)]
    fn from(value: MatrixColumnCount) -> Self {
        match value {
            MatrixColumnCount::Two => Self::Two,
            MatrixColumnCount::Three => Self::Three,
            MatrixColumnCount::Four => Self::Four,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct VectorType(pub ScalarType, pub VectorSize);
impl VectorType {
    pub const fn element_type(&self) -> &ScalarType {
        &self.0
    }

    pub const fn element_count(&self) -> VectorSize {
        self.1
    }

    pub const fn of_matrix(self, count: MatrixColumnCount) -> MatrixType {
        MatrixType(self, count)
    }

    pub const fn matrix_stride(&self) -> Option<u32> {
        match self.0 {
            ScalarType::Bool | ScalarType::Int(32, _) | ScalarType::Float(32) => {
                Some(4 * self.1 as u32)
            }
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum MatrixColumnCount {
    Two = 2,
    Three = 3,
    Four = 4,
}
impl MatrixColumnCount {
    #[inline(always)]
    pub const fn count(self) -> u8 {
        self as _
    }
}
impl TryFrom<u8> for MatrixColumnCount {
    type Error = u8;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            _ => Err(value),
        }
    }
}
impl TryFrom<u32> for MatrixColumnCount {
    type Error = u32;

    #[inline]
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            _ => Err(value),
        }
    }
}
impl From<MatrixColumnCount> for u32 {
    #[inline(always)]
    fn from(value: MatrixColumnCount) -> Self {
        value as _
    }
}
impl From<VectorSize> for MatrixColumnCount {
    #[inline(always)]
    fn from(value: VectorSize) -> Self {
        match value {
            VectorSize::Two => Self::Two,
            VectorSize::Three => Self::Three,
            VectorSize::Four => Self::Four,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct MatrixType(pub VectorType, pub MatrixColumnCount);
impl MatrixType {
    pub const fn element_type(&self) -> &ScalarType {
        self.0.element_type()
    }

    pub const fn column_type(&self) -> &VectorType {
        &self.0
    }

    pub const fn column_count(&self) -> MatrixColumnCount {
        self.1
    }

    pub const fn stride(&self) -> Option<u32> {
        self.0.matrix_stride()
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ImageComponentType {
    Void,
    Scalar(ScalarType),
}
impl From<ScalarType> for ImageComponentType {
    fn from(value: ScalarType) -> Self {
        Self::Scalar(value)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ScalarOrVectorType {
    Scalar(ScalarType),
    Vector(ScalarType, VectorSize),
}
impl ScalarOrVectorType {
    pub const fn scalar(&self) -> &ScalarType {
        match self {
            Self::Scalar(x) => x,
            Self::Vector(x, _) => x,
        }
    }

    pub const fn vector_size(&self) -> Option<VectorSize> {
        match self {
            &Self::Scalar(_) => None,
            &Self::Vector(_, x) => Some(x),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ScalarOrVectorTypeView<'s> {
    Scalar(&'s ScalarType),
    Vector(&'s ScalarType, VectorSize),
}
impl<'s> ScalarOrVectorTypeView<'s> {
    pub const fn scalar(&self) -> &'s ScalarType {
        match self {
            Self::Scalar(x) => x,
            Self::Vector(x, _) => x,
        }
    }

    pub const fn vector_size(&self) -> Option<VectorSize> {
        match self {
            &Self::Scalar(_) => None,
            &Self::Vector(_, x) => Some(x),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct PointerType {
    pub storage_class: super::asm::StorageClass,
    pub base: Type,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Type {
    Void,
    Bool,
    Scalar(ScalarType),
    Vector(ScalarType, VectorSize),
    Matrix(VectorType, MatrixColumnCount),
    Image {
        sampled_type: ImageComponentType,
        dim: super::asm::Dim,
        depth: Option<bool>,
        arrayed: bool,
        multisampled: bool,
        sampled: super::asm::TypeImageSampled,
        image_format: super::asm::ImageFormat,
        access_qualifier: Option<super::asm::AccessQualifier>,
    },
    Sampler,
    SampledImage {
        image_type: Box<Type>,
    },
    Array {
        element_type: Box<Type>,
        length: Box<TypeArrayLength>,
    },
    RuntimeArray {
        element_type: Box<Type>,
    },
    Struct {
        decorations: Vec<super::Decorate>,
        member_types: Vec<TypeStructMember>,
    },
    Opaque {
        name: String,
    },
    Pointer(Box<PointerType>),
    Function {
        return_type: Box<Type>,
        parameter_types: Vec<Type>,
    },
    ForwardPointer {
        pointer_type: Box<Type>,
        storage_class: super::asm::StorageClass,
    },
}
impl From<ScalarType> for Type {
    fn from(value: ScalarType) -> Self {
        Self::Scalar(value)
    }
}
impl From<VectorType> for Type {
    fn from(value: VectorType) -> Self {
        Self::Vector(value.0, value.1)
    }
}
impl From<MatrixType> for Type {
    fn from(value: MatrixType) -> Self {
        Self::Matrix(value.0, value.1)
    }
}
impl From<ImageComponentType> for Type {
    fn from(value: ImageComponentType) -> Self {
        match value {
            ImageComponentType::Void => Self::Void,
            ImageComponentType::Scalar(x) => Self::Scalar(x),
        }
    }
}
impl Type {
    pub const SUBPASS_DATA_IMAGE_TYPE: Self = Self::Image {
        sampled_type: ImageComponentType::Scalar(ScalarType::Float(32)),
        dim: super::asm::Dim::SubpassData,
        depth: Some(false),
        arrayed: false,
        multisampled: false,
        sampled: super::asm::TypeImageSampled::WithReadWriteOps,
        image_format: super::asm::ImageFormat::Unknown,
        access_qualifier: None,
    };

    #[inline(always)]
    pub const fn sint(width: u32) -> Self {
        Self::Scalar(ScalarType::Int(width, true))
    }

    #[inline(always)]
    pub const fn uint(width: u32) -> Self {
        Self::Scalar(ScalarType::Int(width, false))
    }

    #[inline(always)]
    pub const fn float(width: u32) -> Self {
        Self::Scalar(ScalarType::Float(width))
    }

    #[inline(always)]
    pub fn of_vector(self, component_count: u32) -> Self {
        match (self, component_count) {
            (x, 1) => x,
            (Self::Scalar(x), 2) => Self::Vector(x, VectorSize::Two),
            (Self::Scalar(x), 3) => Self::Vector(x, VectorSize::Three),
            (Self::Scalar(x), 4) => Self::Vector(x, VectorSize::Four),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn vector_to_matrix(self, column_count: u32) -> Self {
        match (self, column_count) {
            (x, 1) => x,
            (Self::Vector(x, r), 2) => Self::Matrix(VectorType(x, r), MatrixColumnCount::Two),
            (Self::Vector(x, r), 3) => Self::Matrix(VectorType(x, r), MatrixColumnCount::Three),
            (Self::Vector(x, r), 4) => Self::Matrix(VectorType(x, r), MatrixColumnCount::Four),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn of_matrix(self, row_count: u32, column_count: u32) -> Self {
        match (self, row_count, column_count) {
            (x, 1, 1) => x,
            (Self::Scalar(x), 2, 1) => Self::Vector(x, VectorSize::Two),
            (Self::Scalar(x), 3, 1) => Self::Vector(x, VectorSize::Three),
            (Self::Scalar(x), 4, 1) => Self::Vector(x, VectorSize::Four),
            (Self::Scalar(x), 2, 2) => {
                Self::Matrix(VectorType(x, VectorSize::Two), MatrixColumnCount::Two)
            }
            (Self::Scalar(x), 2, 3) => {
                Self::Matrix(VectorType(x, VectorSize::Two), MatrixColumnCount::Three)
            }
            (Self::Scalar(x), 2, 4) => {
                Self::Matrix(VectorType(x, VectorSize::Two), MatrixColumnCount::Four)
            }
            (Self::Scalar(x), 3, 2) => {
                Self::Matrix(VectorType(x, VectorSize::Three), MatrixColumnCount::Two)
            }
            (Self::Scalar(x), 3, 3) => {
                Self::Matrix(VectorType(x, VectorSize::Three), MatrixColumnCount::Three)
            }
            (Self::Scalar(x), 3, 4) => {
                Self::Matrix(VectorType(x, VectorSize::Three), MatrixColumnCount::Four)
            }
            (Self::Scalar(x), 4, 2) => {
                Self::Matrix(VectorType(x, VectorSize::Four), MatrixColumnCount::Two)
            }
            (Self::Scalar(x), 4, 3) => {
                Self::Matrix(VectorType(x, VectorSize::Four), MatrixColumnCount::Three)
            }
            (Self::Scalar(x), 4, 4) => {
                Self::Matrix(VectorType(x, VectorSize::Four), MatrixColumnCount::Four)
            }
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn of_pointer(self, storage: super::asm::StorageClass) -> PointerType {
        PointerType {
            storage_class: storage,
            base: self,
        }
    }

    #[inline(always)]
    pub fn dereferenced(self) -> Option<Self> {
        match self {
            Self::Pointer(p) => Some(p.base),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn scalar_or_vector(self) -> Option<ScalarOrVectorType> {
        match self {
            Self::Scalar(x) => Some(ScalarOrVectorType::Scalar(x)),
            Self::Vector(x, c) => Some(ScalarOrVectorType::Vector(x, c)),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn scalar_or_vector_view(&self) -> Option<ScalarOrVectorTypeView> {
        match self {
            Self::Scalar(x) => Some(ScalarOrVectorTypeView::Scalar(x)),
            Self::Vector(x, c) => Some(ScalarOrVectorTypeView::Vector(x, *c)),
            _ => None,
        }
    }
}
impl From<PointerType> for Type {
    #[inline(always)]
    fn from(value: PointerType) -> Self {
        Self::Pointer(Box::new(value))
    }
}
