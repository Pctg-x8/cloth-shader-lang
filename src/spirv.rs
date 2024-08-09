use std::io::Write;

pub mod asm;
pub mod types;
pub use self::types::*;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ExecutionModeModifier {
    OriginUpperLeft,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Decorate {
    Block,
    ColMajor,
    MatrixStride(u32),
    Builtin(self::asm::Builtin),
    Location(u32),
    Binding(u32),
    DescriptorSet(u32),
    Offset(u32),
    InputAttachmentIndex(u32),
}
impl Decorate {
    pub fn make_instruction<IdType>(&self, target: IdType) -> Instruction<IdType> {
        match self {
            &Self::Block => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::Block,
                args: vec![],
            },
            &Self::ColMajor => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::ColMajor,
                args: vec![],
            },
            &Self::MatrixStride(s) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::MatrixStride,
                args: vec![s],
            },
            &Self::Builtin(b) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::Builtin,
                args: vec![b as _],
            },
            &Self::Location(loc) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::Location,
                args: vec![loc],
            },
            &Self::Binding(b) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::Binding,
                args: vec![b],
            },
            &Self::DescriptorSet(s) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::DescriptorSet,
                args: vec![s],
            },
            &Self::Offset(o) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::Offset,
                args: vec![o],
            },
            &Self::InputAttachmentIndex(a) => Instruction::Decorate {
                target,
                decoration: self::asm::Decoration::InputAttachmentIndex,
                args: vec![a],
            },
        }
    }

    pub fn make_member_instruction<IdType>(
        &self,
        struct_type: IdType,
        member: u32,
    ) -> Instruction<IdType> {
        match self {
            &Self::Block => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::Block,
                args: vec![],
            },
            &Self::ColMajor => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::ColMajor,
                args: vec![],
            },
            &Self::MatrixStride(s) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::MatrixStride,
                args: vec![s],
            },
            &Self::Builtin(b) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::Builtin,
                args: vec![b as _],
            },
            &Self::Location(loc) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::Location,
                args: vec![loc],
            },
            &Self::Binding(b) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::Binding,
                args: vec![b],
            },
            &Self::DescriptorSet(s) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::DescriptorSet,
                args: vec![s],
            },
            &Self::Offset(o) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::Offset,
                args: vec![o],
            },
            &Self::InputAttachmentIndex(a) => Instruction::MemberDecorate {
                struct_type,
                member,
                decoration: self::asm::Decoration::InputAttachmentIndex,
                args: vec![a],
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Constant {
    True {
        result_type: Type,
    },
    False {
        result_type: Type,
    },
    Constant {
        result_type: Type,
        value_bits: u32,
    },
    Composite {
        result_type: Type,
        constituents: Vec<Constant>,
    },
    Sampler {
        result_type: Type,
        sampler_addressing_mode: self::asm::SamplerAddressingMode,
        normalized: bool,
        sampler_filter_mode: self::asm::SamplerFilterMode,
    },
    Null {
        result_type: Type,
    },
    Undef {
        result_type: Type,
    },
}
impl Constant {
    #[inline]
    pub fn i32vec2(x: i32, y: i32) -> Self {
        Self::Composite {
            result_type: Type::sint(32).of_vector(2),
            constituents: vec![x.into(), y.into()],
        }
    }
}
impl From<u32> for Constant {
    fn from(value: u32) -> Self {
        Self::Constant {
            result_type: Type::uint(32),
            value_bits: value,
        }
    }
}
impl From<i32> for Constant {
    fn from(value: i32) -> Self {
        Self::Constant {
            result_type: Type::sint(32),
            value_bits: unsafe { core::mem::transmute(value) },
        }
    }
}
impl From<f32> for Constant {
    fn from(value: f32) -> Self {
        Self::Constant {
            result_type: Type::float(32),
            value_bits: unsafe { core::mem::transmute(value) },
        }
    }
}

pub type Id = u32;

#[derive(Debug, Clone)]
pub enum Instruction<IdType = Id> {
    Decorate {
        target: IdType,
        decoration: self::asm::Decoration,
        args: Vec<u32>,
    },
    MemberDecorate {
        struct_type: IdType,
        member: u32,
        decoration: self::asm::Decoration,
        args: Vec<u32>,
    },
    ExtInstImport {
        result: IdType,
        name: String,
    },
    ExtInst {
        result_type: IdType,
        result: IdType,
        set: IdType,
        instruction: u32,
        operands: Vec<IdType>,
    },
    MemoryModel {
        addressing_model: self::asm::AddressingModel,
        memory_model: self::asm::MemoryModel,
    },
    EntryPoint {
        execution_model: self::asm::ExecutionModel,
        entry_point: IdType,
        name: String,
        interface: Vec<IdType>,
    },
    ExecutionMode {
        entry_point: IdType,
        mode: self::asm::ExecutionMode,
        args: Vec<u32>,
    },
    Capability {
        capability: self::asm::Capability,
    },
    TypeVoid {
        result: IdType,
    },
    TypeBool {
        result: IdType,
    },
    TypeInt {
        result: IdType,
        width: u32,
        signedness: bool,
    },
    TypeFloat {
        result: IdType,
        width: u32,
    },
    TypeVector {
        result: IdType,
        component_type: IdType,
        component_count: u32,
    },
    TypeMatrix {
        result: IdType,
        column_type: IdType,
        column_count: u32,
    },
    TypeImage {
        result: IdType,
        sampled_type: IdType,
        dim: self::asm::Dim,
        depth: Option<bool>,
        arrayed: bool,
        multisampled: bool,
        sampled: self::asm::TypeImageSampled,
        image_format: self::asm::ImageFormat,
        access_qualifier: Option<self::asm::AccessQualifier>,
    },
    TypeSampler {
        result: IdType,
    },
    TypeSampledImage {
        result: IdType,
        image_type: IdType,
    },
    TypeArray {
        result: IdType,
        element_type: IdType,
        length: IdType,
    },
    TypeRuntimeArray {
        result: IdType,
        element_type: IdType,
    },
    TypeStruct {
        result: IdType,
        member_types: Vec<IdType>,
    },
    TypeOpaque {
        result: IdType,
        name: String,
    },
    TypePointer {
        result: IdType,
        storage_class: self::asm::StorageClass,
        base_type: IdType,
    },
    TypeFunction {
        result: IdType,
        return_type: IdType,
        parameter_types: Vec<IdType>,
    },
    TypeForwardPointer {
        result: IdType,
        pointer_type: IdType,
        storage_class: self::asm::StorageClass,
    },
    ConstantTrue {
        result_type: IdType,
        result: IdType,
    },
    ConstantFalse {
        result_type: IdType,
        result: IdType,
    },
    Constant {
        result_type: IdType,
        result: IdType,
        value_bits: u32,
    },
    ConstantComposite {
        result_type: IdType,
        result: IdType,
        constituents: Vec<IdType>,
    },
    ConstantSampler {
        result_type: IdType,
        result: IdType,
        sampler_addressing_mode: self::asm::SamplerAddressingMode,
        normalized: bool,
        sampler_filter_mode: self::asm::SamplerFilterMode,
    },
    ConstantNull {
        result_type: IdType,
        result: IdType,
    },
    Variable {
        result_type: IdType,
        result: IdType,
        storage_class: self::asm::StorageClass,
        initializer: Option<IdType>,
    },
    Load {
        result_type: IdType,
        result: IdType,
        pointer: IdType,
    },
    Store {
        pointer: IdType,
        object: IdType,
    },
    AccessChain {
        result_type: IdType,
        result: IdType,
        base: IdType,
        indexes: Vec<IdType>,
    },
    Undef {
        result_type: IdType,
        result: IdType,
    },
    Function {
        result_type: IdType,
        result: IdType,
        function_control: self::asm::FunctionControl,
        function_type: IdType,
    },
    FunctionEnd,
    ImageSampleImplicitLod {
        result_type: IdType,
        result: IdType,
        sampled_image: IdType,
        coordinate: IdType,
    },
    ImageRead {
        result_type: IdType,
        result: IdType,
        image: IdType,
        coordinate: IdType,
    },
    ConvertFToU {
        result_type: IdType,
        result: IdType,
        float_value: IdType,
    },
    ConvertFToS {
        result_type: IdType,
        result: IdType,
        float_value: IdType,
    },
    ConvertSToF {
        result_type: IdType,
        result: IdType,
        signed_value: IdType,
    },
    ConvertUToF {
        result_type: IdType,
        result: IdType,
        unsigned_value: IdType,
    },
    VectorShuffle {
        result_type: IdType,
        result: IdType,
        vector1: IdType,
        vector2: IdType,
        components: Vec<u32>,
    },
    CompositeConstruct {
        result_type: IdType,
        result: IdType,
        constituents: Vec<IdType>,
    },
    CompositeExtract {
        result_type: IdType,
        result: IdType,
        composite: IdType,
        indexes: Vec<u32>,
    },
    Transpose {
        result_type: IdType,
        result: IdType,
        matrix: IdType,
    },
    SNegate {
        result_type: IdType,
        result: IdType,
        operand: IdType,
    },
    FNegate {
        result_type: IdType,
        result: IdType,
        operand: IdType,
    },
    IAdd {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FAdd {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    ISub {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FSub {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    IMul {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FMul {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    UDiv {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SDiv {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FDiv {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    UMod {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SRem {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SMod {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FRem {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FMod {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    VectorTimesScalar {
        result_type: IdType,
        result: IdType,
        vector: IdType,
        scalar: IdType,
    },
    MatrixTimesScalar {
        result_type: IdType,
        result: IdType,
        matrix: IdType,
        scalar: IdType,
    },
    VectorTimesMatrix {
        result_type: IdType,
        result: IdType,
        vector: IdType,
        matrix: IdType,
    },
    MatrixTimesVector {
        result_type: IdType,
        result: IdType,
        matrix: IdType,
        vector: IdType,
    },
    MatrixTimesMatrix {
        result_type: IdType,
        result: IdType,
        left_matrix: IdType,
        right_matrix: IdType,
    },
    Dot {
        result_type: IdType,
        result: IdType,
        vector1: IdType,
        vector2: IdType,
    },
    BitwiseOr {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    BitwiseXor {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    BitwiseAnd {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    Not {
        result_type: IdType,
        result: IdType,
        operand: IdType,
    },
    LogicalEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    LogicalNotEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    LogicalOr {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    LogicalAnd {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    LogicalNot {
        result_type: IdType,
        result: IdType,
        operand: IdType,
    },
    Select {
        result_type: IdType,
        result: IdType,
        condition: IdType,
        object1: IdType,
        object2: IdType,
    },
    IEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    INotEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    UGreaterThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SGreaterThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    UGreaterThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SGreaterThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    ULessThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SLessThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    ULessThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    SLessThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdNotEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdLessThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdGreaterThan {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdLessThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    FOrdGreaterThanEqual {
        result_type: IdType,
        result: IdType,
        operand1: IdType,
        operand2: IdType,
    },
    Label {
        result: IdType,
    },
    Return,
    ControlBarrier {
        execution: IdType,
        memory: IdType,
        semantics: IdType,
    },
}
impl<IdType> Instruction<IdType> {
    pub fn relocate<IdType2>(
        self,
        mut relocator: impl FnMut(IdType) -> IdType2,
    ) -> Instruction<IdType2> {
        match self {
            Self::Decorate {
                target,
                decoration,
                args,
            } => Instruction::Decorate {
                target: relocator(target),
                decoration,
                args,
            },
            Self::MemberDecorate {
                struct_type,
                member,
                decoration,
                args,
            } => Instruction::MemberDecorate {
                struct_type: relocator(struct_type),
                member,
                decoration,
                args,
            },
            Self::ExtInstImport { result, name } => Instruction::ExtInstImport {
                result: relocator(result),
                name,
            },
            Self::ExtInst {
                result_type,
                result,
                set,
                instruction,
                operands,
            } => Instruction::ExtInst {
                result_type: relocator(result_type),
                result: relocator(result),
                set: relocator(set),
                instruction,
                operands: operands.into_iter().map(relocator).collect(),
            },
            Self::MemoryModel {
                addressing_model,
                memory_model,
            } => Instruction::MemoryModel {
                addressing_model,
                memory_model,
            },
            Self::EntryPoint {
                execution_model,
                entry_point,
                name,
                interface,
            } => Instruction::EntryPoint {
                execution_model,
                entry_point: relocator(entry_point),
                name,
                interface: interface.into_iter().map(relocator).collect(),
            },
            Self::ExecutionMode {
                entry_point,
                mode,
                args,
            } => Instruction::ExecutionMode {
                entry_point: relocator(entry_point),
                mode,
                args,
            },
            Self::Capability { capability } => Instruction::Capability { capability },
            Self::TypeVoid { result } => Instruction::TypeVoid {
                result: relocator(result),
            },
            Self::TypeBool { result } => Instruction::TypeBool {
                result: relocator(result),
            },
            Self::TypeInt {
                result,
                width,
                signedness,
            } => Instruction::TypeInt {
                result: relocator(result),
                width,
                signedness,
            },
            Self::TypeFloat { result, width } => Instruction::TypeFloat {
                result: relocator(result),
                width,
            },
            Self::TypeVector {
                result,
                component_type,
                component_count,
            } => Instruction::TypeVector {
                result: relocator(result),
                component_type: relocator(component_type),
                component_count,
            },
            Self::TypeMatrix {
                result,
                column_type,
                column_count,
            } => Instruction::TypeMatrix {
                result: relocator(result),
                column_type: relocator(column_type),
                column_count,
            },
            Self::TypeImage {
                result,
                sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier,
            } => Instruction::TypeImage {
                result: relocator(result),
                sampled_type: relocator(sampled_type),
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier,
            },
            Self::TypeSampler { result } => Instruction::TypeSampler {
                result: relocator(result),
            },
            Self::TypeSampledImage { result, image_type } => Instruction::TypeSampledImage {
                result: relocator(result),
                image_type: relocator(image_type),
            },
            Self::TypeArray {
                result,
                element_type,
                length,
            } => Instruction::TypeArray {
                result: relocator(result),
                element_type: relocator(element_type),
                length: relocator(length),
            },
            Self::TypeRuntimeArray {
                result,
                element_type,
            } => Instruction::TypeRuntimeArray {
                result: relocator(result),
                element_type: relocator(element_type),
            },
            Self::TypeStruct {
                result,
                member_types,
            } => Instruction::TypeStruct {
                result: relocator(result),
                member_types: member_types.into_iter().map(relocator).collect(),
            },
            Self::TypeOpaque { result, name } => Instruction::TypeOpaque {
                result: relocator(result),
                name,
            },
            Self::TypePointer {
                result,
                storage_class,
                base_type,
            } => Instruction::TypePointer {
                result: relocator(result),
                storage_class,
                base_type: relocator(base_type),
            },
            Self::TypeFunction {
                result,
                return_type,
                parameter_types,
            } => Instruction::TypeFunction {
                result: relocator(result),
                return_type: relocator(return_type),
                parameter_types: parameter_types.into_iter().map(relocator).collect(),
            },
            Self::TypeForwardPointer {
                result,
                pointer_type,
                storage_class,
            } => Instruction::TypeForwardPointer {
                result: relocator(result),
                pointer_type: relocator(pointer_type),
                storage_class,
            },
            Self::ConstantTrue {
                result_type,
                result,
            } => Instruction::ConstantTrue {
                result_type: relocator(result_type),
                result: relocator(result),
            },
            Self::ConstantFalse {
                result_type,
                result,
            } => Instruction::ConstantFalse {
                result_type: relocator(result_type),
                result: relocator(result),
            },
            Self::Constant {
                result_type,
                result,
                value_bits,
            } => Instruction::Constant {
                result_type: relocator(result_type),
                result: relocator(result),
                value_bits,
            },
            Self::ConstantComposite {
                result_type,
                result,
                constituents,
            } => Instruction::ConstantComposite {
                result_type: relocator(result_type),
                result: relocator(result),
                constituents: constituents.into_iter().map(relocator).collect(),
            },
            Self::ConstantSampler {
                result_type,
                result,
                sampler_addressing_mode,
                normalized,
                sampler_filter_mode,
            } => Instruction::ConstantSampler {
                result_type: relocator(result_type),
                result: relocator(result),
                sampler_addressing_mode,
                normalized,
                sampler_filter_mode,
            },
            Self::ConstantNull {
                result_type,
                result,
            } => Instruction::ConstantNull {
                result_type: relocator(result_type),
                result: relocator(result),
            },
            Self::Variable {
                result_type,
                result,
                storage_class,
                initializer,
            } => Instruction::Variable {
                result_type: relocator(result_type),
                result: relocator(result),
                storage_class,
                initializer: initializer.map(relocator),
            },
            Self::Load {
                result_type,
                result,
                pointer,
            } => Instruction::Load {
                result_type: relocator(result_type),
                result: relocator(result),
                pointer: relocator(pointer),
            },
            Self::Store { pointer, object } => Instruction::Store {
                pointer: relocator(pointer),
                object: relocator(object),
            },
            Self::AccessChain {
                result_type,
                result,
                base,
                indexes,
            } => Instruction::AccessChain {
                result_type: relocator(result_type),
                result: relocator(result),
                base: relocator(base),
                indexes: indexes.into_iter().map(relocator).collect(),
            },
            Self::Undef {
                result_type,
                result,
            } => Instruction::Undef {
                result_type: relocator(result_type),
                result: relocator(result),
            },
            Self::Function {
                result_type,
                result,
                function_control,
                function_type,
            } => Instruction::Function {
                result_type: relocator(result_type),
                result: relocator(result),
                function_control,
                function_type: relocator(function_type),
            },
            Self::FunctionEnd => Instruction::FunctionEnd,
            Self::ImageSampleImplicitLod {
                result_type,
                result,
                sampled_image,
                coordinate,
            } => Instruction::ImageSampleImplicitLod {
                result_type: relocator(result_type),
                result: relocator(result),
                sampled_image: relocator(sampled_image),
                coordinate: relocator(coordinate),
            },
            Self::ImageRead {
                result_type,
                result,
                image,
                coordinate,
            } => Instruction::ImageRead {
                result_type: relocator(result_type),
                result: relocator(result),
                image: relocator(image),
                coordinate: relocator(coordinate),
            },
            Self::ConvertFToU {
                result_type,
                result,
                float_value,
            } => Instruction::ConvertFToU {
                result_type: relocator(result_type),
                result: relocator(result),
                float_value: relocator(float_value),
            },
            Self::ConvertFToS {
                result_type,
                result,
                float_value,
            } => Instruction::ConvertFToS {
                result_type: relocator(result_type),
                result: relocator(result),
                float_value: relocator(float_value),
            },
            Self::ConvertSToF {
                result_type,
                result,
                signed_value,
            } => Instruction::ConvertSToF {
                result_type: relocator(result_type),
                result: relocator(result),
                signed_value: relocator(signed_value),
            },
            Self::ConvertUToF {
                result_type,
                result,
                unsigned_value,
            } => Instruction::ConvertUToF {
                result_type: relocator(result_type),
                result: relocator(result),
                unsigned_value: relocator(unsigned_value),
            },
            Self::VectorShuffle {
                result_type,
                result,
                vector1,
                vector2,
                components,
            } => Instruction::VectorShuffle {
                result_type: relocator(result_type),
                result: relocator(result),
                vector1: relocator(vector1),
                vector2: relocator(vector2),
                components,
            },
            Self::CompositeConstruct {
                result_type,
                result,
                constituents,
            } => Instruction::CompositeConstruct {
                result_type: relocator(result_type),
                result: relocator(result),
                constituents: constituents.into_iter().map(relocator).collect(),
            },
            Self::CompositeExtract {
                result_type,
                result,
                composite,
                indexes,
            } => Instruction::CompositeExtract {
                result_type: relocator(result_type),
                result: relocator(result),
                composite: relocator(composite),
                indexes,
            },
            Self::Transpose {
                result_type,
                result,
                matrix,
            } => Instruction::Transpose {
                result_type: relocator(result_type),
                result: relocator(result),
                matrix: relocator(matrix),
            },
            Self::SNegate {
                result_type,
                result,
                operand,
            } => Instruction::SNegate {
                result_type: relocator(result_type),
                result: relocator(result),
                operand: relocator(operand),
            },
            Self::FNegate {
                result_type,
                result,
                operand,
            } => Instruction::FNegate {
                result_type: relocator(result_type),
                result: relocator(result),
                operand: relocator(operand),
            },
            Self::IAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::IAdd {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FAdd {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::ISub {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::ISub {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FSub {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FSub {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::IMul {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::IMul {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FMul {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FMul {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::UDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::UDiv {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SDiv {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FDiv {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::UMod {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::UMod {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SRem {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SRem {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SMod {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SMod {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FRem {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FRem {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FMod {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FMod {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::VectorTimesScalar {
                result_type,
                result,
                vector,
                scalar,
            } => Instruction::VectorTimesScalar {
                result_type: relocator(result_type),
                result: relocator(result),
                vector: relocator(vector),
                scalar: relocator(scalar),
            },
            Self::MatrixTimesScalar {
                result_type,
                result,
                matrix,
                scalar,
            } => Instruction::MatrixTimesScalar {
                result_type: relocator(result_type),
                result: relocator(result),
                matrix: relocator(matrix),
                scalar: relocator(scalar),
            },
            Self::VectorTimesMatrix {
                result_type,
                result,
                vector,
                matrix,
            } => Instruction::VectorTimesMatrix {
                result_type: relocator(result_type),
                result: relocator(result),
                vector: relocator(vector),
                matrix: relocator(matrix),
            },
            Self::MatrixTimesVector {
                result_type,
                result,
                matrix,
                vector,
            } => Instruction::MatrixTimesVector {
                result_type: relocator(result_type),
                result: relocator(result),
                matrix: relocator(matrix),
                vector: relocator(vector),
            },
            Self::MatrixTimesMatrix {
                result_type,
                result,
                left_matrix,
                right_matrix,
            } => Instruction::MatrixTimesMatrix {
                result_type: relocator(result_type),
                result: relocator(result),
                left_matrix: relocator(left_matrix),
                right_matrix: relocator(right_matrix),
            },
            Self::Dot {
                result_type,
                result,
                vector1,
                vector2,
            } => Instruction::Dot {
                result_type: relocator(result_type),
                result: relocator(result),
                vector1: relocator(vector1),
                vector2: relocator(vector2),
            },
            Self::BitwiseOr {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::BitwiseOr {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::BitwiseXor {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::BitwiseXor {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::BitwiseAnd {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::BitwiseAnd {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::Not {
                result_type,
                result,
                operand,
            } => Instruction::Not {
                result_type: relocator(result_type),
                result: relocator(result),
                operand: relocator(operand),
            },
            Self::LogicalEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::LogicalEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::LogicalNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::LogicalNotEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::LogicalOr {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::LogicalOr {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::LogicalAnd {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::LogicalAnd {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::LogicalNot {
                result_type,
                result,
                operand,
            } => Instruction::LogicalNot {
                result_type: relocator(result_type),
                result: relocator(result),
                operand: relocator(operand),
            },
            Self::Select {
                result_type,
                result,
                condition,
                object1,
                object2,
            } => Instruction::Select {
                result_type: relocator(result_type),
                result: relocator(result),
                condition: relocator(condition),
                object1: relocator(object1),
                object2: relocator(object2),
            },
            Self::IEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::IEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::INotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::INotEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::UGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::UGreaterThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SGreaterThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::UGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::UGreaterThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SGreaterThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::ULessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::ULessThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SLessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SLessThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::ULessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::ULessThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::SLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::SLessThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdNotEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdLessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdLessThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdGreaterThan {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdLessThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::FOrdGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => Instruction::FOrdGreaterThanEqual {
                result_type: relocator(result_type),
                result: relocator(result),
                operand1: relocator(operand1),
                operand2: relocator(operand2),
            },
            Self::Label { result } => Instruction::Label {
                result: relocator(result),
            },
            Self::Return => Instruction::Return,
            Self::ControlBarrier {
                execution,
                memory,
                semantics,
            } => Instruction::ControlBarrier {
                execution: relocator(execution),
                memory: relocator(memory),
                semantics: relocator(semantics),
            },
        }
    }
}
impl Instruction<Id> {
    #[inline(always)]
    const fn opcode(length: u16, opnum: u16) -> u32 {
        ((length as u32) << 16) | opnum as u32
    }

    pub fn serialize_binary<W: BinaryModuleOutputStream + ?Sized>(
        &self,
        w: &mut W,
    ) -> Result<(), W::Error> {
        match self {
            &Self::Decorate {
                target,
                decoration,
                ref args,
            } => {
                w.will_push(3 + args.len())?;
                w.push_array([
                    Self::opcode(3 + args.len() as u16, 71),
                    target,
                    decoration as _,
                ])?;
                w.push_slice(args)?;
                Ok(())
            }
            &Self::MemberDecorate {
                struct_type,
                member,
                decoration,
                ref args,
            } => {
                w.will_push(4 + args.len())?;
                w.push_array([
                    Self::opcode(4 + args.len() as u16, 72),
                    struct_type,
                    member,
                    decoration as _,
                ])?;
                w.push_slice(args)?;
                Ok(())
            }
            &Self::ExtInstImport { result, ref name } => {
                let oplen = 2 + ((name.len() + 1 + 3) >> 2);
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 11), result])?;
                w.push_str(name)?;
                Ok(())
            }
            &Self::ExtInst {
                result_type,
                result,
                set,
                instruction,
                ref operands,
            } => {
                let oplen = 5 + operands.len();
                w.will_push(oplen)?;
                w.push_array([
                    Self::opcode(oplen as _, 12),
                    result_type,
                    result,
                    set,
                    instruction,
                ])?;
                w.push_slice(operands)?;
                Ok(())
            }
            &Self::MemoryModel {
                addressing_model,
                memory_model,
            } => w.push_array([
                Self::opcode(3, 14),
                addressing_model as _,
                memory_model as _,
            ]),
            &Self::EntryPoint {
                execution_model,
                entry_point,
                ref name,
                ref interface,
            } => {
                let oplen = 3 + ((name.len() + 1 + 3) >> 2) + interface.len();
                w.will_push(oplen)?;
                w.push_array([
                    Self::opcode(oplen as _, 15),
                    execution_model as _,
                    entry_point,
                ])?;
                w.push_str(name)?;
                w.push_slice(interface)?;
                Ok(())
            }
            &Self::ExecutionMode {
                entry_point,
                mode,
                ref args,
            } => {
                let oplen = 3 + args.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 16), entry_point, mode as _])?;
                w.push_slice(args)?;
                Ok(())
            }
            &Self::Capability { capability } => {
                w.push_array([Self::opcode(2, 17), capability as _])
            }
            &Self::TypeVoid { result } => w.push_array([Self::opcode(2, 19), result]),
            &Self::TypeBool { result } => w.push_array([Self::opcode(2, 20), result]),
            &Self::TypeInt {
                result,
                width,
                signedness,
            } => w.push_array([
                Self::opcode(4, 21),
                result,
                width,
                if signedness { 1 } else { 0 },
            ]),
            &Self::TypeFloat { result, width } => {
                w.push_array([Self::opcode(3, 22), result, width])
            }
            &Self::TypeVector {
                result,
                component_type,
                component_count,
            } => w.push_array([Self::opcode(4, 23), result, component_type, component_count]),
            &Self::TypeMatrix {
                result,
                column_type,
                column_count,
            } => w.push_array([Self::opcode(4, 24), result, column_type, column_count]),
            &Self::TypeImage {
                result,
                sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier: None,
            } => w.push_array([
                Self::opcode(9, 25),
                result,
                sampled_type,
                dim as _,
                match depth {
                    Some(false) => 0,
                    Some(true) => 1,
                    None => 2,
                },
                if arrayed { 1 } else { 0 },
                if multisampled { 1 } else { 0 },
                sampled as _,
                image_format as _,
            ]),
            &Self::TypeImage {
                result,
                sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier: Some(aq),
            } => w.push_array([
                Self::opcode(10, 25),
                result,
                sampled_type,
                dim as _,
                match depth {
                    Some(false) => 0,
                    Some(true) => 1,
                    None => 2,
                },
                if arrayed { 1 } else { 0 },
                if multisampled { 1 } else { 0 },
                sampled as _,
                image_format as _,
                aq as _,
            ]),
            &Self::TypeSampler { result } => w.push_array([Self::opcode(2, 26), result]),
            &Self::TypeSampledImage { result, image_type } => {
                w.push_array([Self::opcode(3, 27), result, image_type])
            }
            &Self::TypeArray {
                result,
                element_type,
                length,
            } => w.push_array([Self::opcode(4, 28), result, element_type, length]),
            &Self::TypeRuntimeArray {
                result,
                element_type,
            } => w.push_array([Self::opcode(3, 29), result, element_type]),
            &Self::TypeStruct {
                result,
                ref member_types,
            } => {
                let oplen = 2 + member_types.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 30), result])?;
                w.push_slice(member_types)?;
                Ok(())
            }
            &Self::TypeOpaque { result, ref name } => {
                let oplen = 2 + ((name.len() + 1 + 3) >> 2);
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 31), result])?;
                w.push_str(name)?;
                Ok(())
            }
            &Self::TypePointer {
                result,
                storage_class,
                base_type,
            } => w.push_array([Self::opcode(4, 32), result, storage_class as _, base_type]),
            &Self::TypeFunction {
                result,
                return_type,
                ref parameter_types,
            } => {
                let oplen = 3 + parameter_types.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 33), result, return_type])?;
                w.push_slice(parameter_types)?;
                Ok(())
            }
            &Self::TypeForwardPointer {
                result,
                pointer_type,
                storage_class,
            } => w.push_array([
                Self::opcode(3, 39),
                result,
                pointer_type,
                storage_class as _,
            ]),
            &Self::ConstantTrue {
                result_type,
                result,
            } => w.push_array([Self::opcode(3, 41), result_type, result]),
            &Self::ConstantFalse {
                result_type,
                result,
            } => w.push_array([Self::opcode(3, 42), result_type, result]),
            &Self::Constant {
                result_type,
                result,
                value_bits,
            } => w.push_array([Self::opcode(4, 43), result_type, result, value_bits]),
            &Self::ConstantComposite {
                result_type,
                result,
                ref constituents,
            } => {
                let oplen = 3 + constituents.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 44), result_type, result])?;
                w.push_slice(constituents)?;
                Ok(())
            }
            &Self::ConstantSampler {
                result_type,
                result,
                sampler_addressing_mode,
                normalized,
                sampler_filter_mode,
            } => w.push_array([
                Self::opcode(6, 45),
                result_type,
                result,
                sampler_addressing_mode as _,
                if normalized { 1 } else { 0 },
                sampler_filter_mode as _,
            ]),
            &Self::ConstantNull {
                result_type,
                result,
            } => w.push_array([Self::opcode(3, 46), result_type, result]),
            &Self::Variable {
                result_type,
                result,
                storage_class,
                initializer: None,
            } => w.push_array([Self::opcode(4, 59), result_type, result, storage_class as _]),
            &Self::Variable {
                result_type,
                result,
                storage_class,
                initializer: Some(init),
            } => w.push_array([
                Self::opcode(5, 59),
                result_type,
                result,
                storage_class as _,
                init,
            ]),
            &Self::Load {
                result_type,
                result,
                pointer,
            } => w.push_array([Self::opcode(4, 61), result_type, result, pointer]),
            &Self::Store { pointer, object } => {
                w.push_array([Self::opcode(3, 62), pointer, object])
            }
            &Self::AccessChain {
                result_type,
                result,
                base,
                ref indexes,
            } => {
                let oplen = 4 + indexes.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 65), result_type, result, base])?;
                w.push_slice(indexes)?;
                Ok(())
            }
            &Self::Undef {
                result_type,
                result,
            } => w.push_array([Self::opcode(3, 1), result_type, result]),
            &Self::Function {
                result_type,
                result,
                function_control,
                function_type,
            } => w.push_array([
                Self::opcode(5, 54),
                result_type,
                result,
                function_control.bits(),
                function_type,
            ]),
            &Self::FunctionEnd => w.push(Self::opcode(1, 56)),
            &Self::ImageSampleImplicitLod {
                result_type,
                result,
                sampled_image,
                coordinate,
            } => w.push_array([
                Self::opcode(5, 87),
                result_type,
                result,
                sampled_image,
                coordinate,
            ]),
            &Self::ImageRead {
                result_type,
                result,
                image,
                coordinate,
            } => w.push_array([Self::opcode(5, 98), result_type, result, image, coordinate]),
            &Self::ConvertFToU {
                result_type,
                result,
                float_value,
            } => w.push_array([Self::opcode(4, 109), result_type, result, float_value]),
            &Self::ConvertFToS {
                result_type,
                result,
                float_value,
            } => w.push_array([Self::opcode(4, 110), result_type, result, float_value]),
            &Self::ConvertSToF {
                result_type,
                result,
                signed_value,
            } => w.push_array([Self::opcode(4, 111), result_type, result, signed_value]),
            &Self::ConvertUToF {
                result_type,
                result,
                unsigned_value,
            } => w.push_array([Self::opcode(4, 112), result_type, result, unsigned_value]),
            &Self::VectorShuffle {
                result_type,
                result,
                vector1,
                vector2,
                ref components,
            } => {
                let oplen = 5 + components.len();
                w.will_push(oplen)?;
                w.push_array([
                    Self::opcode(oplen as _, 79),
                    result_type,
                    result,
                    vector1,
                    vector2,
                ])?;
                w.push_slice(components)?;
                Ok(())
            }
            &Self::CompositeConstruct {
                result_type,
                result,
                ref constituents,
            } => {
                let oplen = 3 + constituents.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 80), result_type, result])?;
                w.push_slice(constituents)?;
                Ok(())
            }
            &Self::CompositeExtract {
                result_type,
                result,
                composite,
                ref indexes,
            } => {
                let oplen = 4 + indexes.len();
                w.will_push(oplen)?;
                w.push_array([Self::opcode(oplen as _, 81), result_type, result, composite])?;
                w.push_slice(indexes)?;
                Ok(())
            }
            &Self::Transpose {
                result_type,
                result,
                matrix,
            } => w.push_array([Self::opcode(4, 84), result_type, result, matrix]),
            &Self::SNegate {
                result_type,
                result,
                operand,
            } => w.push_array([Self::opcode(4, 126), result_type, result, operand]),
            &Self::FNegate {
                result_type,
                result,
                operand,
            } => w.push_array([Self::opcode(4, 127), result_type, result, operand]),
            &Self::IAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 128),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 129),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::ISub {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 130),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FSub {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 131),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::IMul {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 132),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FMul {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 133),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::UDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 134),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 135),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FDiv {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 136),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::UMod {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 137),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SRem {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 138),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SMod {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 139),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FRem {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 140),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FMod {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 141),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::VectorTimesScalar {
                result_type,
                result,
                vector,
                scalar,
            } => w.push_array([Self::opcode(5, 142), result_type, result, vector, scalar]),
            &Self::MatrixTimesScalar {
                result_type,
                result,
                matrix,
                scalar,
            } => w.push_array([Self::opcode(5, 143), result_type, result, matrix, scalar]),
            &Self::VectorTimesMatrix {
                result_type,
                result,
                vector,
                matrix,
            } => w.push_array([Self::opcode(5, 144), result_type, result, vector, matrix]),
            &Self::MatrixTimesVector {
                result_type,
                result,
                matrix,
                vector,
            } => w.push_array([Self::opcode(5, 145), result_type, result, matrix, vector]),
            &Self::MatrixTimesMatrix {
                result_type,
                result,
                left_matrix,
                right_matrix,
            } => w.push_array([
                Self::opcode(5, 146),
                result_type,
                result,
                left_matrix,
                right_matrix,
            ]),
            &Self::Dot {
                result_type,
                result,
                vector1,
                vector2,
            } => w.push_array([Self::opcode(5, 148), result_type, result, vector1, vector2]),
            &Self::BitwiseOr {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 197),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::BitwiseXor {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 198),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::BitwiseAnd {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 199),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::Not {
                result_type,
                result,
                operand,
            } => w.push_array([Self::opcode(4, 200), result_type, result, operand]),
            &Self::LogicalEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 164),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::LogicalNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 165),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::LogicalOr {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 166),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::LogicalAnd {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 167),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::LogicalNot {
                result_type,
                result,
                operand,
            } => w.push_array([Self::opcode(4, 168), result_type, result, operand]),
            &Self::Select {
                result_type,
                result,
                condition,
                object1,
                object2,
            } => w.push_array([
                Self::opcode(6, 169),
                result_type,
                result,
                condition,
                object1,
                object2,
            ]),
            &Self::IEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 170),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::INotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 171),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::UGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 172),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 173),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::UGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 174),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 175),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::ULessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 176),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SLessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 177),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::ULessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 178),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::SLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 179),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 180),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 182),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdLessThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 184),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 186),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 188),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::FOrdGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            } => w.push_array([
                Self::opcode(5, 190),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            &Self::Label { result } => w.push_array([Self::opcode(2, 248), result]),
            &Self::Return => w.push(Self::opcode(1, 253)),
            &Self::ControlBarrier {
                execution,
                memory,
                semantics,
            } => w.push_array([Self::opcode(4, 224), execution, memory, semantics]),
        }
    }
}

pub struct BinaryModuleHeader {
    pub magic_number: u32,
    pub major_version: u8,
    pub minor_version: u8,
    pub generator_magic_number: u32,
    pub bound: u32,
}
impl BinaryModuleHeader {
    pub const MAGIC_NUMBER: u32 = 0x07230203;

    pub fn serialize<W: BinaryModuleOutputStream>(&self, writer: &mut W) -> Result<(), W::Error> {
        writer.push_array([
            self.magic_number,
            (self.major_version as u32) << 16 | (self.minor_version as u32) << 8,
            self.generator_magic_number,
            self.bound,
            0,
        ])
    }
}

pub trait BinaryModuleOutputStream {
    type Error;

    #[allow(unused_variables)]
    fn will_push(&mut self, blocks: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    fn push(&mut self, value: u32) -> Result<(), Self::Error>;
    fn push_slice(&mut self, values: &[u32]) -> Result<(), Self::Error>;
    fn push_vec(&mut self, values: Vec<u32>) -> Result<(), Self::Error> {
        self.push_slice(&values)
    }
    fn push_array<const N: usize>(&mut self, values: [u32; N]) -> Result<(), Self::Error> {
        self.push_slice(&values)
    }
    fn push_str(&mut self, s: &str) -> Result<(), Self::Error> {
        let mut blocks = s
            .as_bytes()
            .chunks(4)
            .map(|x| {
                u32::from_ne_bytes(match x {
                    &[a, b, c, d] => [a, b, c, d],
                    &[a, b, c] => [a, b, c, 0x00],
                    &[a, b] => [a, b, 0x00, 0x00],
                    &[a] => [a, 0x00, 0x00, 0x00],
                    _ => unreachable!(),
                })
            })
            .collect::<Vec<_>>();
        if blocks.last().map_or(true, |x| x.to_ne_bytes()[3] != 0x00) {
            // add zero terminated block
            blocks.push(0);
        }

        self.push_vec(blocks)
    }
}
impl<W: std::io::Write> BinaryModuleOutputStream for std::io::BufWriter<W> {
    type Error = std::io::Error;

    fn push(&mut self, value: u32) -> Result<(), Self::Error> {
        self.write_all(&value.to_ne_bytes())
    }
    fn push_slice(&mut self, values: &[u32]) -> Result<(), Self::Error> {
        self.write_all(unsafe {
            core::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() << 2)
        })
    }
}
