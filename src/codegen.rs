use std::collections::{HashMap, HashSet};

use entrypoint::ShaderEntryPointDescription;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::{
        expr::{ConstModifiers, SimplifiedExpression},
        ExprRef,
    },
    ref_path::RefPath,
    spirv as spv,
};

pub mod entrypoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpvSectionLocalId {
    ExtInstImport(spv::Id),
    TypeConst(spv::Id),
    GlobalVariable(spv::Id),
    Function(spv::Id),
    CurrentFunction(spv::Id),
}

pub struct SpvModuleEmissionContext {
    pub capabilities: HashSet<spv::Capability>,
    pub ext_inst_imports: HashMap<String, SpvSectionLocalId>,
    pub addressing_model: spv::AddressingModel,
    pub memory_model: spv::MemoryModel,
    pub entry_point_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub execution_mode_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub annotation_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub type_const_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub global_variable_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub function_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub latest_ext_inst_import_id: spv::Id,
    pub latest_type_const_id: spv::Id,
    pub latest_global_variable_id: spv::Id,
    pub latest_function_id: spv::Id,
    pub defined_const_map: HashMap<spv::Constant, SpvSectionLocalId>,
    pub defined_type_map: HashMap<spv::Type, SpvSectionLocalId>,
}
impl SpvModuleEmissionContext {
    pub fn new() -> Self {
        Self {
            capabilities: HashSet::new(),
            ext_inst_imports: HashMap::new(),
            addressing_model: spv::AddressingModel::Logical,
            memory_model: spv::MemoryModel::GLSL450,
            entry_point_ops: Vec::new(),
            execution_mode_ops: Vec::new(),
            annotation_ops: Vec::new(),
            type_const_ops: Vec::new(),
            global_variable_ops: Vec::new(),
            function_ops: Vec::new(),
            latest_ext_inst_import_id: 0,
            latest_type_const_id: 0,
            latest_global_variable_id: 0,
            latest_function_id: 0,
            defined_const_map: HashMap::new(),
            defined_type_map: HashMap::new(),
        }
    }

    pub fn serialize_ops(self) -> (Vec<spv::Instruction>, spv::Id) {
        let ext_inst_import_offset = 1;
        let type_const_offset = ext_inst_import_offset + self.latest_ext_inst_import_id;
        let global_variable_offset = type_const_offset + self.latest_type_const_id;
        let function_offset = global_variable_offset + self.latest_global_variable_id;

        let mut sorted_ext_inst_imports = self
            .ext_inst_imports
            .into_iter()
            .map(|(name, id)| (id, name))
            .collect::<Vec<_>>();
        sorted_ext_inst_imports.sort_by_key(|&(id, _)| id);

        let capabilities = self
            .capabilities
            .into_iter()
            .map(|c| spv::Instruction::Capability { capability: c });
        let ext_inst_imports = sorted_ext_inst_imports
            .into_iter()
            .map(|(id, name)| spv::Instruction::ExtInstImport { result: id, name });

        let mut max_id = 1;
        let ops = capabilities
            .chain(ext_inst_imports)
            .chain(std::iter::once(spv::Instruction::MemoryModel {
                addressing_model: self.addressing_model,
                memory_model: self.memory_model,
            }))
            .chain(self.entry_point_ops.into_iter())
            .chain(self.execution_mode_ops.into_iter())
            .chain(self.annotation_ops.into_iter())
            .chain(self.type_const_ops.into_iter())
            .chain(self.global_variable_ops.into_iter())
            .chain(self.function_ops.into_iter())
            .map(|x| {
                x.relocate(|id| {
                    let nid = match id {
                        SpvSectionLocalId::ExtInstImport(x) => x + ext_inst_import_offset,
                        SpvSectionLocalId::TypeConst(x) => x + type_const_offset,
                        SpvSectionLocalId::GlobalVariable(x) => x + global_variable_offset,
                        SpvSectionLocalId::Function(x) => x + function_offset,
                        SpvSectionLocalId::CurrentFunction(_) => unreachable!(),
                    };
                    max_id = max_id.max(nid);
                    nid
                })
            })
            .collect();
        (ops, max_id)
    }

    pub fn request_ext_inst_set(&mut self, name: String) -> SpvSectionLocalId {
        *self.ext_inst_imports.entry(name).or_insert_with(|| {
            self.latest_ext_inst_import_id += 1;

            SpvSectionLocalId::ExtInstImport(self.latest_ext_inst_import_id - 1)
        })
    }

    pub fn new_type_const_id(&mut self) -> SpvSectionLocalId {
        self.latest_type_const_id += 1;

        SpvSectionLocalId::TypeConst(self.latest_type_const_id - 1)
    }

    pub fn new_global_variable_id(&mut self) -> SpvSectionLocalId {
        self.latest_global_variable_id += 1;

        SpvSectionLocalId::GlobalVariable(self.latest_global_variable_id - 1)
    }

    pub fn new_function_id(&mut self) -> SpvSectionLocalId {
        self.latest_function_id += 1;

        SpvSectionLocalId::Function(self.latest_function_id - 1)
    }

    pub fn decorate(&mut self, target: SpvSectionLocalId, decorations: &[spv::Decorate]) {
        self.annotation_ops
            .extend(decorations.iter().map(|d| match d {
                spv::Decorate::DescriptorSet(loc) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::DescriptorSet,
                    args: vec![*loc],
                },
                spv::Decorate::Binding(b) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::Binding,
                    args: vec![*b],
                },
                spv::Decorate::InputAttachmentIndex(b) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::InputAttachmentIndex,
                    args: vec![*b],
                },
                spv::Decorate::Block => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::Block,
                    args: vec![],
                },
                spv::Decorate::Location(loc) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::Location,
                    args: vec![*loc],
                },
                spv::Decorate::Builtin(b) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::Builtin,
                    args: vec![*b as _],
                },
                spv::Decorate::Offset(o) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::Offset,
                    args: vec![*o],
                },
                spv::Decorate::ColMajor => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::ColMajor,
                    args: Vec::new(),
                },
                spv::Decorate::MatrixStride(stride) => spv::Instruction::Decorate {
                    target,
                    decoration: spv::Decoration::MatrixStride,
                    args: vec![*stride],
                },
            }));
    }

    pub fn declare_global_variable(
        &mut self,
        ty: spv::Type,
        storage_class: spv::StorageClass,
    ) -> SpvSectionLocalId {
        let ptr_ty = self.request_type_id(ty.of_pointer(storage_class));
        let id = self.new_global_variable_id();
        self.global_variable_ops.push(spv::Instruction::Variable {
            result_type: ptr_ty,
            result: id,
            storage_class,
            initializer: None,
        });

        id
    }

    pub fn request_const_id(&mut self, c: spv::Constant) -> SpvSectionLocalId {
        if let Some(id) = self.defined_const_map.get(&c) {
            return *id;
        }

        let id = match c {
            spv::Constant::True { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantTrue {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::False { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantFalse {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::Constant {
                ref result_type,
                value_bits,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::Constant {
                    result_type,
                    result: id,
                    value_bits,
                });
                id
            }
            spv::Constant::Composite {
                ref result_type,
                ref constituents,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let constituents = constituents
                    .iter()
                    .map(|x| self.request_const_id(x.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::ConstantComposite {
                        result_type,
                        result: id,
                        constituents,
                    });
                id
            }
            spv::Constant::Sampler {
                ref result_type,
                sampler_addressing_mode,
                normalized,
                sampler_filter_mode,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantSampler {
                    result_type,
                    result: id,
                    sampler_addressing_mode,
                    normalized,
                    sampler_filter_mode,
                });
                id
            }
            spv::Constant::Null { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantNull {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::Undef { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::Undef {
                    result_type,
                    result: id,
                });
                id
            }
        };

        self.defined_const_map.insert(c, id);
        id
    }

    pub fn request_type_id(&mut self, t: spv::Type) -> SpvSectionLocalId {
        if let Some(id) = self.defined_type_map.get(&t) {
            return *id;
        }

        match t {
            spv::Type::Void => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeVoid { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Bool => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeBool { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Int { width, signedness } => {
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeInt {
                    result: id,
                    width,
                    signedness,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Float { width } => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeFloat { result: id, width });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Vector {
                ref component_type,
                component_count,
            } => {
                let component_type = self.request_type_id(*component_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeVector {
                    result: id,
                    component_type,
                    component_count,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Matrix {
                ref column_type,
                column_count,
            } => {
                let column_type = self.request_type_id(*column_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeMatrix {
                    result: id,
                    column_type,
                    column_count,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Image {
                ref sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier,
            } => {
                let sampled_type = self.request_type_id(*sampled_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeImage {
                    result: id,
                    sampled_type,
                    dim,
                    depth,
                    arrayed,
                    multisampled,
                    sampled,
                    image_format,
                    access_qualifier,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Sampler => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeSampler { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::SampledImage { ref image_type } => {
                let image_type = self.request_type_id(*image_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeSampledImage {
                        result: id,
                        image_type,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Array {
                ref element_type,
                ref length,
            } => {
                let element_type = self.request_type_id(*element_type.clone());
                let id = self.new_type_const_id();
                let length = match **length {
                    spv::TypeArrayLength::SpecConstantID(x) => SpvSectionLocalId::TypeConst(x),
                    spv::TypeArrayLength::ConstExpr(ref x) => self.request_const_id(x.clone()),
                };

                self.type_const_ops.push(spv::Instruction::TypeArray {
                    result: id,
                    element_type,
                    length,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::RuntimeArray { ref element_type } => {
                let element_type = self.request_type_id(*element_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeRuntimeArray {
                        result: id,
                        element_type,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Struct {
                ref member_types,
                ref decorations,
            } => {
                let member_type_ids = member_types
                    .iter()
                    .map(|x| self.request_type_id(x.ty.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeStruct {
                    result: id,
                    member_types: member_type_ids,
                });
                self.decorate(id, decorations);

                for (n, x) in member_types.iter().enumerate() {
                    self.annotation_ops
                        .extend(x.decorations.iter().map(|d| match d {
                            spv::Decorate::Offset(x) => spv::Instruction::MemberDecorate {
                                struct_type: id,
                                member: n as _,
                                decoration: spv::Decoration::Offset,
                                args: vec![*x],
                            },
                            spv::Decorate::ColMajor => spv::Instruction::MemberDecorate {
                                struct_type: id,
                                member: n as _,
                                decoration: spv::Decoration::ColMajor,
                                args: Vec::new(),
                            },
                            spv::Decorate::MatrixStride(stride) => {
                                spv::Instruction::MemberDecorate {
                                    struct_type: id,
                                    member: n as _,
                                    decoration: spv::Decoration::MatrixStride,
                                    args: vec![*stride],
                                }
                            }
                            _ => unimplemented!(),
                        }));
                }

                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Opaque { ref name } => {
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeOpaque {
                    result: id,
                    name: name.clone(),
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Pointer {
                storage_class,
                ref base_type,
            } => {
                let base_type = self.request_type_id(*base_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypePointer {
                    result: id,
                    storage_class,
                    base_type,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Function {
                ref return_type,
                ref parameter_types,
            } => {
                let return_type = self.request_type_id(*return_type.clone());
                let parameter_types = parameter_types
                    .iter()
                    .map(|x| self.request_type_id(x.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeFunction {
                    result: id,
                    return_type,
                    parameter_types,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::ForwardPointer {
                ref pointer_type,
                storage_class,
            } => {
                let pointer_type = self.request_type_id(*pointer_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeForwardPointer {
                        result: id,
                        pointer_type,
                        storage_class,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
        }
    }
}

pub struct SpvFunctionBodyEmissionContext<'m> {
    pub module_ctx: &'m mut SpvModuleEmissionContext,
    pub entry_point_maps: ShaderEntryPointMaps,
    pub ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub latest_id: spv::Id,
    pub emitted_expression_id: HashMap<usize, Option<(SpvSectionLocalId, spv::Type)>>,
}
impl<'s, 'm> SpvFunctionBodyEmissionContext<'m> {
    pub fn new(module_ctx: &'m mut SpvModuleEmissionContext, maps: ShaderEntryPointMaps) -> Self {
        Self {
            module_ctx,
            entry_point_maps: maps,
            ops: Vec::new(),
            latest_id: 0,
            emitted_expression_id: HashMap::new(),
        }
    }

    pub fn new_id(&mut self) -> SpvSectionLocalId {
        self.latest_id += 1;

        SpvSectionLocalId::CurrentFunction(self.latest_id - 1)
    }

    #[inline(always)]
    fn issue_typed_ids(&mut self, ty: spv::Type) -> (SpvSectionLocalId, SpvSectionLocalId) {
        (self.module_ctx.request_type_id(ty), self.new_id())
    }

    #[inline]
    pub fn select(
        &mut self,
        output_type: spv::Type,
        condition: SpvSectionLocalId,
        object1: SpvSectionLocalId,
        object2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let (result_type, result) = self.issue_typed_ids(output_type.clone());
        self.ops.push(spv::Instruction::Select {
            result_type,
            result,
            condition,
            object1,
            object2,
        });
        (result, output_type)
    }

    #[inline]
    fn log_and(
        &mut self,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::LogicalAnd {
            result_type,
            result,
            operand1,
            operand2,
        });
        (result, rt)
    }

    #[inline]
    fn log_or(
        &mut self,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::LogicalOr {
            result_type,
            result,
            operand1,
            operand2,
        });
        (result, rt)
    }

    #[inline]
    fn equal(
        &mut self,
        operand_class: EqCompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            EqCompareOperandClass::Bool => spv::Instruction::LogicalEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            EqCompareOperandClass::Int => spv::Instruction::IEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            EqCompareOperandClass::Float => spv::Instruction::FOrdEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    fn not_equal(
        &mut self,
        operand_class: EqCompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            EqCompareOperandClass::Bool => spv::Instruction::LogicalNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            EqCompareOperandClass::Int => spv::Instruction::INotEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            EqCompareOperandClass::Float => spv::Instruction::FOrdNotEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    fn less_than(
        &mut self,
        operand_class: CompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            CompareOperandClass::SInt => spv::Instruction::SLessThan {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::UInt => spv::Instruction::ULessThan {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::Float => spv::Instruction::FOrdLessThan {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    fn less_than_eq(
        &mut self,
        operand_class: CompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            CompareOperandClass::SInt => spv::Instruction::SLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::UInt => spv::Instruction::ULessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::Float => spv::Instruction::FOrdLessThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    fn greater_than(
        &mut self,
        operand_class: CompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            CompareOperandClass::SInt => spv::Instruction::SGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::UInt => spv::Instruction::UGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::Float => spv::Instruction::FOrdGreaterThan {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    fn greater_than_eq(
        &mut self,
        operand_class: CompareOperandClass,
        component_count: u32,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::Bool.of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        let op = match operand_class {
            CompareOperandClass::SInt => spv::Instruction::SGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::UInt => spv::Instruction::UGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
            CompareOperandClass::Float => spv::Instruction::FOrdGreaterThanEqual {
                result_type,
                result,
                operand1,
                operand2,
            },
        };
        self.ops.push(op);
        (result, rt)
    }

    #[inline]
    pub fn iadd(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::IAdd {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn fadd(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FAdd {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn isub(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::ISub {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn fsub(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FSub {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn imul(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::IMul {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn fmul(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FMul {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn vector_times_scalar(
        &mut self,
        output_ty: spv::Type,
        vector: SpvSectionLocalId,
        scalar: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::VectorTimesScalar {
            result_type,
            result,
            vector,
            scalar,
        });
        result
    }

    #[inline]
    pub fn matrix_times_scalar(
        &mut self,
        output_ty: spv::Type,
        matrix: SpvSectionLocalId,
        scalar: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::MatrixTimesScalar {
            result_type,
            result,
            matrix,
            scalar,
        });
        result
    }

    #[inline]
    pub fn matrix_times_vector(
        &mut self,
        output_ty: spv::Type,
        matrix: SpvSectionLocalId,
        vector: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::MatrixTimesVector {
            result_type,
            result,
            matrix,
            vector,
        });
        result
    }

    #[inline]
    pub fn vector_times_matrix(
        &mut self,
        output_ty: spv::Type,
        vector: SpvSectionLocalId,
        matrix: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::VectorTimesMatrix {
            result_type,
            result,
            vector,
            matrix,
        });
        result
    }

    #[inline]
    pub fn matrix_times_matrix(
        &mut self,
        output_ty: spv::Type,
        left_matrix: SpvSectionLocalId,
        right_matrix: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::MatrixTimesMatrix {
            result_type,
            result,
            left_matrix,
            right_matrix,
        });
        result
    }

    #[inline]
    pub fn sdiv(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::SDiv {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn udiv(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::UDiv {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn fdiv(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FDiv {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn srem(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::SRem {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn umod(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::UMod {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn frem(
        &mut self,
        output_ty: spv::Type,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FRem {
            result_type,
            result,
            operand1,
            operand2,
        });
        result
    }

    #[inline]
    pub fn snegate(
        &mut self,
        output_ty: spv::Type,
        operand: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::SNegate {
            result_type,
            result,
            operand,
        });
        result
    }

    #[inline]
    pub fn fnegate(
        &mut self,
        output_ty: spv::Type,
        operand: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_ty);
        self.ops.push(spv::Instruction::FNegate {
            result_type,
            result,
            operand,
        });
        result
    }

    #[inline]
    pub fn convert_sint_to_float(
        &mut self,
        component_count: u32,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::float(32).of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::ConvertSToF {
            result_type,
            result,
            signed_value: input,
        });
        (result, rt)
    }

    #[inline]
    pub fn convert_uint_to_float(
        &mut self,
        component_count: u32,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::float(32).of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::ConvertUToF {
            result_type,
            result,
            unsigned_value: input,
        });
        (result, rt)
    }

    #[inline]
    pub fn convert_float_to_sint(
        &mut self,
        component_count: u32,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::sint(32).of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::ConvertFToS {
            result_type,
            result,
            float_value: input,
        });
        (result, rt)
    }

    #[inline]
    pub fn convert_float_to_uint(
        &mut self,
        component_count: u32,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt = spv::Type::uint(32).of_vector(component_count);
        let (result_type, result) = self.issue_typed_ids(rt.clone());
        self.ops.push(spv::Instruction::ConvertFToU {
            result_type,
            result,
            float_value: input,
        });
        (result, rt)
    }

    #[inline]
    pub fn composite_extract(
        &mut self,
        output_type: spv::Type,
        source: SpvSectionLocalId,
        indices: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type);
        self.ops.push(spv::Instruction::CompositeExtract {
            result_type,
            result,
            composite: source,
            indexes: indices,
        });
        result
    }

    #[inline]
    pub fn vector_shuffle(
        &mut self,
        output_type: spv::Type,
        source1: SpvSectionLocalId,
        source2: SpvSectionLocalId,
        components: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type);
        self.ops.push(spv::Instruction::VectorShuffle {
            result_type,
            result,
            vector1: source1,
            vector2: source2,
            components,
        });
        result
    }

    #[inline]
    pub fn vector_shuffle_1(
        &mut self,
        output_type: spv::Type,
        source: SpvSectionLocalId,
        components: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type);
        self.ops.push(spv::Instruction::VectorShuffle {
            result_type,
            result,
            vector1: source,
            vector2: source,
            components,
        });
        result
    }

    #[inline]
    pub fn load(
        &mut self,
        output_type: spv::Type,
        pointer: SpvSectionLocalId,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type);
        self.ops.push(spv::Instruction::Load {
            result_type,
            result,
            pointer,
        });
        result
    }

    #[inline]
    pub fn access_chain(
        &mut self,
        output_type: spv::Type,
        root_ptr: SpvSectionLocalId,
        indices: Vec<SpvSectionLocalId>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type);
        self.ops.push(spv::Instruction::AccessChain {
            result_type,
            result,
            base: root_ptr,
            indexes: indices,
        });
        result
    }

    #[inline]
    pub fn chained_load(
        &mut self,
        output_type: spv::Type,
        storage_class: spv::StorageClass,
        root_ptr: SpvSectionLocalId,
        indices: Vec<SpvSectionLocalId>,
    ) -> SpvSectionLocalId {
        let ptr = self.access_chain(
            output_type.clone().of_pointer(storage_class),
            root_ptr,
            indices,
        );
        self.load(output_type, ptr)
    }
}

#[derive(Clone, Copy)]
enum EqCompareOperandClass {
    Bool,
    Int,
    Float,
}
impl EqCompareOperandClass {
    const fn of(t: &spv::Type) -> Self {
        match t {
            spv::Type::Bool => Self::Bool,
            spv::Type::Int { .. } => Self::Int,
            spv::Type::Float { .. } => Self::Float,
            _ => unreachable!(),
        }
    }
}
#[derive(Clone, Copy)]
enum CompareOperandClass {
    UInt,
    SInt,
    Float,
}
impl CompareOperandClass {
    const fn of(t: &spv::Type) -> Self {
        match t {
            spv::Type::Int {
                signedness: true, ..
            } => Self::SInt,
            spv::Type::Int {
                signedness: false, ..
            } => Self::UInt,
            spv::Type::Float { .. } => Self::Float,
            _ => unreachable!(),
        }
    }
}

enum GlobalAccessType {
    Direct(SpvSectionLocalId, spv::Type),
    PushConstantStruct {
        struct_var: SpvSectionLocalId,
        member_index: u32,
        member_ty: spv::Type,
    },
}

pub struct ShaderEntryPointMaps {
    refpath_to_global_var: HashMap<RefPath, GlobalAccessType>,
    output_global_vars: Vec<SpvSectionLocalId>,
    interface_global_vars: Vec<SpvSectionLocalId>,
}
impl ShaderEntryPointMaps {
    pub fn iter_interface_global_vars<'s>(
        &'s self,
    ) -> impl Iterator<Item = &'s SpvSectionLocalId> + 's {
        self.interface_global_vars.iter()
    }
}
pub fn emit_entry_point_spv_ops<'s>(
    ep: &ShaderEntryPointDescription<'s>,
    ctx: &mut SpvModuleEmissionContext,
) -> ShaderEntryPointMaps {
    let mut entry_point_maps = ShaderEntryPointMaps {
        refpath_to_global_var: HashMap::new(),
        output_global_vars: Vec::new(),
        interface_global_vars: Vec::new(),
    };

    for a in ep.output_variables.iter() {
        let gvid = ctx.declare_global_variable(a.ty.clone(), spv::StorageClass::Output);
        ctx.decorate(gvid, &a.decorations);

        entry_point_maps.output_global_vars.push(gvid);
        entry_point_maps.interface_global_vars.push(gvid);
    }

    for a in ep.input_variables.iter() {
        let vid = ctx.declare_global_variable(a.ty.clone(), spv::StorageClass::Input);
        ctx.decorate(vid, &a.decorations);

        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(vid, a.ty.clone()),
        );
        entry_point_maps.interface_global_vars.push(vid);
    }

    for a in ep.uniform_variables.iter() {
        let vid = ctx.declare_global_variable(a.ty.clone(), spv::StorageClass::Uniform);
        ctx.decorate(vid, &a.decorations);

        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(vid, a.ty.clone()),
        );
        entry_point_maps.interface_global_vars.push(vid);
    }

    if !ep.push_constant_variables.is_empty() {
        let block_ty = spv::Type::Struct {
            decorations: vec![spv::Decorate::Block],
            member_types: ep
                .push_constant_variables
                .iter()
                .map(|x| {
                    let mut decorations = vec![spv::Decorate::Offset(x.offset)];
                    if let spv::Type::Matrix {
                        ref column_type, ..
                    } = x.ty
                    {
                        decorations.extend([
                            spv::Decorate::ColMajor,
                            spv::Decorate::MatrixStride(column_type.matrix_stride().unwrap()),
                        ]);
                    }

                    spv::TypeStructMember {
                        ty: x.ty.clone(),
                        decorations,
                    }
                })
                .collect(),
        };
        let block_tid = ctx.request_type_id(block_ty.clone());
        let block_var = ctx.declare_global_variable(block_ty, spv::StorageClass::PushConstant);
        for (n, a) in ep.push_constant_variables.iter().enumerate() {
            entry_point_maps.refpath_to_global_var.insert(
                a.original_refpath.clone(),
                GlobalAccessType::PushConstantStruct {
                    struct_var: block_var,
                    member_index: n as _,
                    member_ty: a.ty.clone(),
                },
            );
        }
        entry_point_maps.interface_global_vars.push(block_var);
    }

    entry_point_maps
}

pub fn emit_function_body_spv_ops(
    body: &[(SimplifiedExpression, ConcreteType)],
    returning: ExprRef,
    ctx: &mut SpvFunctionBodyEmissionContext,
) {
    for (n, (b, _)) in body.iter().enumerate() {
        if !b.is_pure() {
            emit_expr_spv_ops(body, n, ctx);
        }
    }
}

fn emit_expr_spv_ops(
    body: &[(SimplifiedExpression, ConcreteType)],
    expr_id: usize,
    ctx: &mut SpvFunctionBodyEmissionContext,
) -> Option<(SpvSectionLocalId, spv::Type)> {
    if let Some(r) = ctx.emitted_expression_id.get(&expr_id) {
        return r.clone();
    }

    let result = match &body[expr_id].0 {
        SimplifiedExpression::Select(c, t, e) => {
            let (c, ct) = emit_expr_spv_ops(body, c.0, ctx).unwrap();

            let requires_control_flow_branching = ct != spv::Type::Bool
                || matches!(
                    (&body[t.0].0, &body[e.0].0),
                    (SimplifiedExpression::ScopedBlock { .. }, _)
                        | (_, SimplifiedExpression::ScopedBlock { .. })
                );

            if requires_control_flow_branching {
                // impure select
                unimplemented!("Control Flow Branching strategy");
            } else {
                let (t, tt) = emit_expr_spv_ops(body, t.0, ctx).unwrap();
                let (e, et) = emit_expr_spv_ops(body, e.0, ctx).unwrap();
                assert_eq!(tt, et);

                Some(ctx.select(tt.clone(), c, t, e))
            }
        }
        SimplifiedExpression::ConstructIntrinsicComposite(it, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|&x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
                .unzip();

            match it {
                IntrinsicType::Float4 => {
                    assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

                    let rt = spv::Type::float(32).of_vector(4);
                    let result_type = ctx.module_ctx.request_type_id(rt.clone());
                    let is_constant = args
                        .iter()
                        .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

                    let result_id;
                    if is_constant {
                        result_id = ctx.module_ctx.new_type_const_id();
                        ctx.module_ctx
                            .type_const_ops
                            .push(spv::Instruction::ConstantComposite {
                                result_type,
                                result: result_id,
                                constituents: match args.len() {
                                    1 => args.repeat(4),
                                    2 => args.repeat(2),
                                    4 => args,
                                    _ => panic!("Error: component count mismatching"),
                                },
                            });
                    } else {
                        result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::CompositeConstruct {
                            result_type,
                            result: result_id,
                            constituents: match args.len() {
                                1 => args.repeat(4),
                                2 => args.repeat(2),
                                4 => args,
                                _ => panic!("Error: component count mismatching"),
                            },
                        });
                    };

                    Some((result_id, rt))
                }
                IntrinsicType::Float3 => {
                    assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

                    let rt = spv::Type::float(32).of_vector(3);
                    let result_type = ctx.module_ctx.request_type_id(rt.clone());
                    let is_constant = args
                        .iter()
                        .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

                    let result_id;
                    if is_constant {
                        result_id = ctx.module_ctx.new_type_const_id();
                        ctx.module_ctx
                            .type_const_ops
                            .push(spv::Instruction::ConstantComposite {
                                result_type,
                                result: result_id,
                                constituents: match args.len() {
                                    1 => args.repeat(3),
                                    3 => args,
                                    _ => panic!("Error: component count mismatching"),
                                },
                            });
                    } else {
                        result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::CompositeConstruct {
                            result_type,
                            result: result_id,
                            constituents: match args.len() {
                                1 => args.repeat(3),
                                3 => args,
                                _ => panic!("Error: component count mismatching"),
                            },
                        });
                    };

                    Some((result_id, rt))
                }
                _ => unimplemented!("Composite construction for {it:?}"),
            }
        }
        SimplifiedExpression::LogAnd(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let component_count = match lt {
                spv::Type::Bool => 1,
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Bool => component_count,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            Some(ctx.log_and(component_count, l, r))
        }
        SimplifiedExpression::LogOr(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let component_count = match lt {
                spv::Type::Bool => 1,
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Bool => component_count,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            Some(ctx.log_or(component_count, l, r))
        }
        SimplifiedExpression::Eq(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (EqCompareOperandClass::of(component_type), component_count),
                ref x => (EqCompareOperandClass::of(x), 1),
            };

            Some(ctx.equal(cls, cc, l, r))
        }
        SimplifiedExpression::Ne(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (EqCompareOperandClass::of(component_type), component_count),
                ref x => (EqCompareOperandClass::of(x), 1),
            };

            Some(ctx.not_equal(cls, cc, l, r))
        }
        SimplifiedExpression::Lt(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (CompareOperandClass::of(component_type), component_count),
                ref x => (CompareOperandClass::of(x), 1),
            };

            Some(ctx.less_than(cls, cc, l, r))
        }
        SimplifiedExpression::Le(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (CompareOperandClass::of(component_type), component_count),
                ref x => (CompareOperandClass::of(x), 1),
            };

            Some(ctx.less_than_eq(cls, cc, l, r))
        }
        SimplifiedExpression::Gt(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (CompareOperandClass::of(component_type), component_count),
                ref x => (CompareOperandClass::of(x), 1),
            };

            Some(ctx.greater_than(cls, cc, l, r))
        }
        SimplifiedExpression::Ge(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let (cls, cc) = match lt {
                spv::Type::Vector {
                    ref component_type,
                    component_count,
                } => (CompareOperandClass::of(component_type), component_count),
                ref x => (CompareOperandClass::of(x), 1),
            };

            Some(ctx.greater_than_eq(cls, cc, l, r))
        }
        SimplifiedExpression::BitAnd(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);
            assert!(
                matches!(lt, spv::Type::Int { .. })
                    || matches!(lt, spv::Type::Vector { ref component_type, .. } if matches!(&**component_type, spv::Type::Int {..}))
            );

            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
            ctx.ops.push(spv::Instruction::BitwiseAnd {
                result_type,
                result,
                operand1: l,
                operand2: r,
            });
            Some((result, lt))
        }
        SimplifiedExpression::BitOr(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);
            assert!(
                matches!(lt, spv::Type::Int { .. })
                    || matches!(lt, spv::Type::Vector { ref component_type, .. } if matches!(&**component_type, spv::Type::Int {..}))
            );

            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
            ctx.ops.push(spv::Instruction::BitwiseOr {
                result_type,
                result,
                operand1: l,
                operand2: r,
            });
            Some((result, lt))
        }
        SimplifiedExpression::BitXor(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);
            assert!(
                matches!(lt, spv::Type::Int { .. })
                    || matches!(lt, spv::Type::Vector { ref component_type, .. } if matches!(&**component_type, spv::Type::Int {..}))
            );

            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
            ctx.ops.push(spv::Instruction::BitwiseXor {
                result_type,
                result,
                operand1: l,
                operand2: r,
            });
            Some((result, lt))
        }
        SimplifiedExpression::Add(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            Some((
                match lt {
                    spv::Type::Int { .. } => ctx.iadd(lt.clone(), l, r),
                    spv::Type::Float { .. } => ctx.fadd(lt.clone(), l, r),
                    spv::Type::Vector {
                        ref component_type, ..
                    } => match &**component_type {
                        spv::Type::Int { .. } => ctx.iadd(lt.clone(), l, r),
                        spv::Type::Float { .. } => ctx.fadd(lt.clone(), l, r),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                lt,
            ))
        }
        SimplifiedExpression::Sub(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            Some((
                match lt {
                    spv::Type::Int { .. } => ctx.isub(lt.clone(), l, r),
                    spv::Type::Float { .. } => ctx.fsub(lt.clone(), l, r),
                    spv::Type::Vector {
                        ref component_type, ..
                    } => match &**component_type {
                        spv::Type::Int { .. } => ctx.isub(lt.clone(), l, r),
                        spv::Type::Float { .. } => ctx.fsub(lt.clone(), l, r),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                lt,
            ))
        }
        SimplifiedExpression::Mul(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();

            match (&lt, &rt) {
                (
                    &spv::Type::Vector {
                        ref component_type,
                        component_count,
                    },
                    &spv::Type::Float { width },
                ) if **component_type == spv::Type::float(width) => {
                    let rt = spv::Type::float(width).of_vector(component_count);
                    Some((ctx.vector_times_scalar(rt.clone(), l, r), rt))
                }
                (
                    &spv::Type::Matrix {
                        ref column_type,
                        column_count,
                    },
                    &spv::Type::Float { width },
                ) => match **column_type {
                    spv::Type::Vector {
                        ref component_type,
                        component_count,
                    } if **component_type == spv::Type::float(width) => {
                        let rt = spv::Type::float(width).of_matrix(component_count, column_count);
                        Some((ctx.matrix_times_scalar(rt.clone(), l, r), rt))
                    }
                    _ => unreachable!(),
                },
                (
                    &spv::Type::Matrix {
                        column_type: ref ctl,
                        column_count: ccl,
                    },
                    &spv::Type::Vector {
                        component_type: ref ctr,
                        component_count: ccr,
                    },
                ) => match **ctl {
                    spv::Type::Vector {
                        ref component_type,
                        component_count,
                    } => match **component_type {
                        spv::Type::Float { width } if **ctr == spv::Type::float(width) => {
                            let rt = spv::Type::float(width).of_vector(component_count);
                            Some((ctx.matrix_times_vector(rt.clone(), l, r), rt))
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                (
                    &spv::Type::Vector {
                        component_type: ref ctl,
                        component_count: ccl,
                    },
                    &spv::Type::Matrix {
                        column_type: ref ctr,
                        column_count: ccr,
                    },
                ) => match **ctr {
                    spv::Type::Vector {
                        ref component_type,
                        component_count,
                    } => match **component_type {
                        spv::Type::Float { width } if **ctl == spv::Type::float(width) => {
                            let rt = spv::Type::float(width).of_vector(ccr);
                            Some((ctx.vector_times_matrix(rt.clone(), l, r), rt))
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                (
                    &spv::Type::Matrix {
                        column_type: ref ctl,
                        column_count: ccl,
                    },
                    &spv::Type::Matrix {
                        column_type: ref ctr,
                        column_count: ccr,
                    },
                ) if **ctl == **ctr => match **ctl {
                    spv::Type::Vector {
                        ref component_type,
                        component_count,
                    } => match **component_type {
                        spv::Type::Float { width } => {
                            let rt = spv::Type::float(width).of_matrix(component_count, ccr);
                            Some((ctx.matrix_times_matrix(rt.clone(), l, r), rt))
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                (a, b) if a == b => Some((
                    match &lt {
                        spv::Type::Int { .. } => ctx.imul(lt.clone(), l, r),
                        spv::Type::Float { .. } => ctx.fmul(lt.clone(), l, r),
                        spv::Type::Vector {
                            ref component_type, ..
                        } => match &**component_type {
                            spv::Type::Int { .. } => ctx.imul(lt.clone(), l, r),
                            spv::Type::Float { .. } => ctx.fmul(lt.clone(), l, r),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    },
                    lt,
                )),
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Div(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            Some((
                match lt {
                    spv::Type::Int {
                        signedness: true, ..
                    } => ctx.sdiv(lt.clone(), l, r),
                    spv::Type::Int {
                        signedness: false, ..
                    } => ctx.udiv(lt.clone(), l, r),
                    spv::Type::Float { .. } => ctx.fdiv(lt.clone(), l, r),
                    spv::Type::Vector {
                        ref component_type, ..
                    } => match &**component_type {
                        spv::Type::Int {
                            signedness: true, ..
                        } => ctx.sdiv(lt.clone(), l, r),
                        spv::Type::Int {
                            signedness: false, ..
                        } => ctx.udiv(lt.clone(), l, r),
                        spv::Type::Float { .. } => ctx.fdiv(lt.clone(), l, r),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                lt,
            ))
        }
        SimplifiedExpression::Rem(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            Some((
                match lt {
                    spv::Type::Int {
                        signedness: true, ..
                    } => ctx.srem(lt.clone(), l, r),
                    spv::Type::Int {
                        signedness: false, ..
                    } => ctx.umod(lt.clone(), l, r),
                    spv::Type::Float { .. } => ctx.frem(lt.clone(), l, r),
                    spv::Type::Vector {
                        ref component_type, ..
                    } => match &**component_type {
                        spv::Type::Int {
                            signedness: true, ..
                        } => ctx.srem(lt.clone(), l, r),
                        spv::Type::Int {
                            signedness: false, ..
                        } => ctx.umod(lt.clone(), l, r),
                        spv::Type::Float { .. } => ctx.frem(lt.clone(), l, r),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                lt,
            ))
        }
        SimplifiedExpression::Pow(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
            assert_eq!(lt, rt);

            let rt = match lt {
                spv::Type::Float { width } => spv::Type::float(width),
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Float { width } => {
                        spv::Type::float(width).of_vector(component_count)
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            let glsl_std_450_lib_set_id =
                ctx.module_ctx.request_ext_inst_set("GLSL.std.450".into());
            let (result_type, result) = ctx.issue_typed_ids(rt.clone());
            ctx.ops.push(spv::Instruction::ExtInst {
                result_type,
                result,
                set: glsl_std_450_lib_set_id,
                instruction: 26,
                operands: vec![l, r],
            });
            Some((result, rt))
        }
        SimplifiedExpression::Neg(x) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();

            Some((
                match &xt {
                    &spv::Type::Int {
                        signedness: true, ..
                    } => ctx.snegate(xt.clone(), x),
                    &spv::Type::Float { .. } => ctx.fnegate(xt.clone(), x),
                    &spv::Type::Vector {
                        ref component_type, ..
                    } => match &**component_type {
                        &spv::Type::Int {
                            signedness: true, ..
                        } => ctx.snegate(xt.clone(), x),
                        &spv::Type::Float { .. } => ctx.fnegate(xt.clone(), x),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                xt,
            ))
        }
        SimplifiedExpression::Cast(x, t) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();

            Some(match xt {
                spv::Type::Int {
                    width: 32,
                    signedness: true,
                } => match t {
                    ConcreteType::Intrinsic(IntrinsicType::Float) => {
                        ctx.convert_sint_to_float(1, x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Int {
                    width: 32,
                    signedness: false,
                } => match t {
                    ConcreteType::Intrinsic(IntrinsicType::Float) => {
                        ctx.convert_uint_to_float(1, x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Float { width: 32 } => match t {
                    ConcreteType::Intrinsic(IntrinsicType::SInt) => ctx.convert_float_to_sint(1, x),
                    ConcreteType::Intrinsic(IntrinsicType::UInt) => ctx.convert_float_to_uint(1, x),
                    _ => unreachable!(),
                },
                spv::Type::Vector {
                    component_type,
                    component_count: 2,
                } => match *component_type {
                    spv::Type::Int {
                        width: 32,
                        signedness: true,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float2) => {
                            ctx.convert_sint_to_float(2, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Int {
                        width: 32,
                        signedness: false,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float2) => {
                            ctx.convert_uint_to_float(2, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Float { width: 32 } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::SInt2) => {
                            ctx.convert_float_to_sint(2, x)
                        }
                        ConcreteType::Intrinsic(IntrinsicType::UInt2) => {
                            ctx.convert_float_to_uint(2, x)
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                spv::Type::Vector {
                    component_type,
                    component_count: 3,
                } => match *component_type {
                    spv::Type::Int {
                        width: 32,
                        signedness: true,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float3) => {
                            ctx.convert_sint_to_float(3, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Int {
                        width: 32,
                        signedness: false,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float3) => {
                            ctx.convert_uint_to_float(3, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Float { width: 32 } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::SInt3) => {
                            ctx.convert_float_to_sint(3, x)
                        }
                        ConcreteType::Intrinsic(IntrinsicType::UInt3) => {
                            ctx.convert_float_to_uint(3, x)
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                spv::Type::Vector {
                    component_type,
                    component_count: 4,
                } => match *component_type {
                    spv::Type::Int {
                        width: 32,
                        signedness: true,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float4) => {
                            ctx.convert_sint_to_float(4, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Int {
                        width: 32,
                        signedness: false,
                    } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::Float4) => {
                            ctx.convert_uint_to_float(4, x)
                        }
                        _ => unreachable!(),
                    },
                    spv::Type::Float { width: 32 } => match t {
                        ConcreteType::Intrinsic(IntrinsicType::SInt4) => {
                            ctx.convert_float_to_sint(4, x)
                        }
                        ConcreteType::Intrinsic(IntrinsicType::UInt4) => {
                            ctx.convert_float_to_uint(4, x)
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            })
        }
        &SimplifiedExpression::Swizzle1(x, a) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            Some((
                ctx.composite_extract(*component_type.clone(), x, vec![a as _]),
                *component_type,
            ))
        }
        &SimplifiedExpression::Swizzle2(x, a, b) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = component_type.of_vector(2);
            Some((
                ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _]),
                rt,
            ))
        }
        &SimplifiedExpression::Swizzle3(x, a, b, c) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = component_type.of_vector(3);
            Some((
                ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _]),
                rt,
            ))
        }
        &SimplifiedExpression::Swizzle4(x, a, b, c, d) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = component_type.of_vector(4);
            Some((
                ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _, d as _]),
                rt,
            ))
        }
        &SimplifiedExpression::VectorShuffle4(v1, v2, a, b, c, d) => {
            let (v1, v1t) = emit_expr_spv_ops(body, v1.0, ctx).unwrap();
            let (v2, _) = emit_expr_spv_ops(body, v2.0, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = v1t else {
                unreachable!("v1t = {v1t:?}");
            };

            let rt = component_type.of_vector(4);
            Some((
                ctx.vector_shuffle(rt.clone(), v1, v2, vec![a as _, b as _, c as _, d as _]),
                rt,
            ))
        }
        SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
            match ctx.entry_point_maps.refpath_to_global_var.get(rp) {
                Some(GlobalAccessType::Direct(gv, vty)) => {
                    let (gv, vty) = (gv.clone(), vty.clone());

                    Some((ctx.load(vty.clone(), gv), vty))
                }
                Some(&GlobalAccessType::PushConstantStruct {
                    struct_var,
                    member_index,
                    ref member_ty,
                }) => {
                    let member_ty = member_ty.clone();

                    let ac_index_id = ctx.module_ctx.request_const_id(member_index.into());
                    Some((
                        ctx.chained_load(
                            member_ty.clone(),
                            spv::StorageClass::PushConstant,
                            struct_var,
                            vec![ac_index_id],
                        ),
                        member_ty,
                    ))
                }
                // TODO: あとで再帰組む
                None => match rp {
                    RefPath::Member(root, index) => {
                        Some(match ctx.entry_point_maps.refpath_to_global_var.get(root) {
                            Some(GlobalAccessType::Direct(gv, vty)) => {
                                let (gv, vty) = (gv.clone(), vty.clone());
                                let member_ty = match vty {
                                    spv::Type::Struct { member_types, .. } => {
                                        member_types[*index].ty.clone()
                                    }
                                    _ => unreachable!("cannot member ref"),
                                };

                                let ac_index_id =
                                    ctx.module_ctx.request_const_id((*index as u32).into());
                                // TODO: Uniform決め打ちは後で何とかする（Storage Bufferサポートするときに詰む）
                                (
                                    ctx.chained_load(
                                        member_ty.clone(),
                                        spv::StorageClass::Uniform,
                                        gv,
                                        vec![ac_index_id],
                                    ),
                                    member_ty,
                                )
                            }
                            Some(&GlobalAccessType::PushConstantStruct {
                                struct_var,
                                member_index,
                                ref member_ty,
                            }) => {
                                let member_ty = member_ty.clone();
                                let final_ty = match member_ty {
                                    spv::Type::Struct { member_types, .. } => {
                                        member_types[*index].ty.clone()
                                    }
                                    _ => unreachable!("cannot member ref"),
                                };

                                let ac_index_id =
                                    ctx.module_ctx.request_const_id(member_index.into());
                                let ac_index2_id =
                                    ctx.module_ctx.request_const_id((*index as u32).into());
                                (
                                    ctx.chained_load(
                                        final_ty.clone(),
                                        spv::StorageClass::PushConstant,
                                        struct_var,
                                        vec![ac_index_id, ac_index2_id],
                                    ),
                                    final_ty,
                                )
                            }
                            None => panic!("no corresponding canonical refpath found? {rp:?}"),
                        })
                    }
                    _ => panic!("no corresponding canonical refpath found? {rp:?}"),
                },
            }
        }
        SimplifiedExpression::ConstSInt(s, mods) => {
            let mut x: i32 = match s
                .0
                .slice
                .strip_prefix("0x")
                .or_else(|| s.0.slice.strip_prefix("0X"))
            {
                Some(left) => i32::from_str_radix(left, 16).unwrap(),
                None => s.0.slice.parse().unwrap(),
            };
            if mods.contains(ConstModifiers::NEGATE) {
                x = -x;
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = !x;
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x == 0 { 1 } else { 0 };
            }

            Some((
                ctx.module_ctx.request_const_id(x.into()),
                spv::Type::sint(32),
            ))
        }
        SimplifiedExpression::ConstUInt(s, mods) => {
            let mut x: u32 = match s
                .0
                .slice
                .strip_prefix("0x")
                .or_else(|| s.0.slice.strip_prefix("0X"))
            {
                Some(left) => u32::from_str_radix(left, 16).unwrap(),
                None => s.0.slice.parse().unwrap(),
            };
            if mods.contains(ConstModifiers::NEGATE) {
                panic!("applying negate to unsigned number?");
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = !x;
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x == 0 { 1 } else { 0 };
            }

            Some((
                ctx.module_ctx.request_const_id(x.into()),
                spv::Type::uint(32),
            ))
        }
        SimplifiedExpression::ConstFloat(s, mods) => {
            let mut x: f32 =
                s.0.slice
                    .strip_suffix(['f', 'F'])
                    .unwrap_or(s.0.slice)
                    .parse()
                    .unwrap();
            if mods.contains(ConstModifiers::NEGATE) {
                x = -x;
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = unsafe { core::mem::transmute(!core::mem::transmute::<_, u32>(x)) };
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x != 0.0 { 0.0 } else { 1.0 };
            }

            Some((
                ctx.module_ctx.request_const_id(x.into()),
                spv::Type::float(32),
            ))
        }
        SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.SubpassLoad", _, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
                .unzip();
            assert!(arg_ty.len() == 1 && arg_ty[0] == spv::Type::subpass_data_image_type());

            let rt = spv::Type::float(32).of_vector(4);
            let result_type = ctx.module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            let coordinate_const = ctx
                .module_ctx
                .request_const_id(spv::Constant::i32vec2(0, 0));
            ctx.ops.push(spv::Instruction::ImageRead {
                result_type,
                result: result_id,
                image: args[0],
                coordinate: coordinate_const,
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Dot#Float3", _, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
                .unzip();
            assert!(
                arg_ty.len() == 2
                    && arg_ty
                        .iter()
                        .all(|t| *t == spv::Type::float(32).of_vector(3))
            );

            let rt = spv::Type::float(32);
            let (result_type, result) = ctx.issue_typed_ids(rt.clone());
            ctx.ops.push(spv::Instruction::Dot {
                result_type,
                result,
                vector1: args[0],
                vector2: args[1],
            });
            Some((result, rt))
        }
        SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Normalize#Float3", _, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
                .unzip();
            assert_eq!(arg_ty.len(), 1);
            assert_eq!(arg_ty[0], spv::Type::float(32).of_vector(3));

            let glsl_std_450_ext_set = ctx.module_ctx.request_ext_inst_set("GLSL.std.450".into());

            let rt = spv::Type::float(32).of_vector(3);
            let (result_type, result) = ctx.issue_typed_ids(rt.clone());
            ctx.ops.push(spv::Instruction::ExtInst {
                result_type,
                result,
                set: glsl_std_450_ext_set,
                instruction: 69,
                operands: args,
            });
            Some((result, rt))
        }
        SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Transpose#Float4x4", _, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
                .unzip();
            assert_eq!(arg_ty.len(), 1);
            assert_eq!(arg_ty[0], spv::Type::float(32).of_matrix(4, 4));

            let rt = spv::Type::float(32).of_matrix(4, 4);
            let (result_type, result) = ctx.issue_typed_ids(rt.clone());
            ctx.ops.push(spv::Instruction::Transpose {
                result_type,
                result,
                matrix: args[0],
            });
            Some((result, rt))
        }
        SimplifiedExpression::StoreOutput(x, o) => {
            let (x, _xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
            ctx.ops.push(spv::Instruction::Store {
                pointer: ctx.entry_point_maps.output_global_vars[*o],
                object: x,
            });

            None
        }
        SimplifiedExpression::ScopedBlock {
            expressions,
            returning,
            ..
        } => emit_expr_spv_ops(expressions, returning.0, ctx),
        x => unimplemented!("{x:?}"),
    };

    ctx.emitted_expression_id.insert(expr_id, result.clone());
    result
}
