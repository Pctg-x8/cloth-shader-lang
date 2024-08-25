use std::collections::{HashMap, HashSet};

use entrypoint::ShaderEntryPointDescription;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::{
        block::{
            Block, BlockFlowInstruction, BlockInstruction, BlockRef, IntrinsicBinaryOperation,
            IntrinsicUnaryOperation, RegisterRef,
        },
        expr::{ConstModifiers, SimplifiedExpression},
        ExprRef, FunctionBody,
    },
    ref_path::RefPath,
    scope::SymbolScope,
    spirv as spv,
    symbol::{
        meta::{BuiltinInputOutput, SymbolAttribute},
        UserDefinedFunctionSymbol,
    },
    utils::roundup2,
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
    pub capabilities: HashSet<spv::asm::Capability>,
    pub ext_inst_imports: HashMap<String, SpvSectionLocalId>,
    pub addressing_model: spv::asm::AddressingModel,
    pub memory_model: spv::asm::MemoryModel,
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
    pub defined_descriptor_set_bound_global_vars:
        HashMap<(u32, u32, spv::Type, spv::asm::StorageClass), SpvSectionLocalId>,
    pub defined_push_constant_global_vars: HashMap<spv::Type, SpvSectionLocalId>,
}
impl SpvModuleEmissionContext {
    pub fn new() -> Self {
        Self {
            capabilities: HashSet::new(),
            ext_inst_imports: HashMap::new(),
            addressing_model: spv::asm::AddressingModel::Logical,
            memory_model: spv::asm::MemoryModel::GLSL450,
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
            defined_descriptor_set_bound_global_vars: HashMap::new(),
            defined_push_constant_global_vars: HashMap::new(),
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
            .extend(decorations.iter().map(|d| d.make_instruction(target)));
    }

    pub fn declare_global_variable(
        &mut self,
        ty: spv::Type,
        storage_class: spv::asm::StorageClass,
    ) -> SpvSectionLocalId {
        let ptr_ty = self.request_type_id(ty.of_pointer(storage_class).into());
        let id = self.new_global_variable_id();
        self.global_variable_ops.push(spv::Instruction::Variable {
            result_type: ptr_ty,
            result: id,
            storage_class,
            initializer: None,
        });

        id
    }

    pub fn request_descriptor_set_bound_global_var(
        &mut self,
        set: u32,
        binding: u32,
        ty: spv::Type,
        storage_class: spv::asm::StorageClass,
    ) -> SpvSectionLocalId {
        let k = (set, binding, ty, storage_class);
        if let Some(&v) = self.defined_descriptor_set_bound_global_vars.get(&k) {
            return v;
        }

        let id = self.declare_global_variable(k.2.clone(), k.3);
        self.defined_descriptor_set_bound_global_vars.insert(k, id);
        id
    }

    pub fn request_push_constant_global_var(&mut self, ty: spv::Type) -> SpvSectionLocalId {
        if let Some(&v) = self.defined_push_constant_global_vars.get(&ty) {
            return v;
        }

        let id = self.declare_global_variable(ty.clone(), spv::asm::StorageClass::PushConstant);
        self.defined_push_constant_global_vars.insert(ty, id);
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
            spv::Type::Scalar(spv::ScalarType::Bool) => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeBool { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Scalar(spv::ScalarType::Int(width, signedness)) => {
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeInt {
                    result: id,
                    width,
                    signedness,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Scalar(spv::ScalarType::Float(width)) => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeFloat { result: id, width });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Vector(ref component_type, component_count) => {
                let component_type = self.request_type_id(component_type.clone().into());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeVector {
                    result: id,
                    component_type,
                    component_count: component_count as _,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Matrix(ref column_type, column_count) => {
                let column_type = self.request_type_id(column_type.clone().into());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeMatrix {
                    result: id,
                    column_type,
                    column_count: column_count as _,
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
                let sampled_type = self.request_type_id(sampled_type.clone().into());
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

                self.annotation_ops
                    .extend(member_types.iter().enumerate().flat_map(|(n, x)| {
                        x.decorations
                            .iter()
                            .map(move |d| d.make_member_instruction(id, n as _))
                    }));

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
            spv::Type::Pointer(ref p) => {
                let base_type = self.request_type_id(p.base.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypePointer {
                    result: id,
                    storage_class: p.storage_class,
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
    pub ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub latest_id: spv::Id,
    pub emitted_expression_id: HashMap<usize, Option<(SpvSectionLocalId, spv::Type)>>,
}
impl<'s, 'm> SpvFunctionBodyEmissionContext<'m> {
    pub fn new(module_ctx: &'m mut SpvModuleEmissionContext) -> Self {
        Self {
            module_ctx,
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        operand1: SpvSectionLocalId,
        operand2: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            Some(cc) => spv::ScalarType::Bool.of_vector(cc).into(),
            None => spv::ScalarType::Bool.into(),
        };
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
        component_count: Option<spv::VectorSize>,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            None => spv::ScalarType::Float(32).into(),
            Some(c) => spv::ScalarType::Float(32).of_vector(c).into(),
        };
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
        component_count: Option<spv::VectorSize>,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            None => spv::ScalarType::Float(32).into(),
            Some(c) => spv::ScalarType::Float(32).of_vector(c).into(),
        };
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
        component_count: Option<spv::VectorSize>,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            None => spv::ScalarType::Int(32, true).into(),
            Some(c) => spv::ScalarType::Int(32, true).of_vector(c).into(),
        };
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
        component_count: Option<spv::VectorSize>,
        input: SpvSectionLocalId,
    ) -> (SpvSectionLocalId, spv::Type) {
        let rt: spv::Type = match component_count {
            None => spv::ScalarType::Int(32, false).into(),
            Some(c) => spv::ScalarType::Int(32, false).of_vector(c).into(),
        };
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
        output_type: spv::ScalarType,
        source: SpvSectionLocalId,
        indices: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type.into());
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
        output_type: spv::VectorType,
        source1: SpvSectionLocalId,
        source2: SpvSectionLocalId,
        components: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type.into());
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
        output_type: spv::VectorType,
        source: SpvSectionLocalId,
        components: Vec<u32>,
    ) -> SpvSectionLocalId {
        let (result_type, result) = self.issue_typed_ids(output_type.into());
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
        storage_class: spv::asm::StorageClass,
        root_ptr: SpvSectionLocalId,
        indices: Vec<SpvSectionLocalId>,
    ) -> SpvSectionLocalId {
        let ptr = self.access_chain(
            output_type.clone().of_pointer(storage_class).into(),
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
    const fn of(t: &spv::ScalarType) -> Self {
        match t {
            spv::ScalarType::Bool => Self::Bool,
            spv::ScalarType::Int(_, _) => Self::Int,
            spv::ScalarType::Float(_) => Self::Float,
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
    const fn of(t: &spv::ScalarType) -> Self {
        match t {
            spv::ScalarType::Int(_, true) => Self::SInt,
            spv::ScalarType::Int(_, false) => Self::UInt,
            spv::ScalarType::Float(_) => Self::Float,
            spv::ScalarType::Bool => unreachable!(),
        }
    }
}

enum GlobalAccessType {
    Direct(SpvSectionLocalId, spv::Type, spv::asm::StorageClass),
    PushConstantStruct {
        struct_var: SpvSectionLocalId,
        member_index: u32,
        member_ty: spv::Type,
    },
}

fn spv_type(c: &ConcreteType, symbol_attr: &SymbolAttribute) -> spv::Type {
    match c {
        ConcreteType::Intrinsic(IntrinsicType::Unit) => spv::Type::Void,
        ConcreteType::Intrinsic(IntrinsicType::Bool) => spv::Type::Bool,
        ConcreteType::Intrinsic(IntrinsicType::SInt) => spv::Type::sint(32),
        ConcreteType::Intrinsic(IntrinsicType::UInt) => spv::Type::uint(32),
        ConcreteType::Intrinsic(IntrinsicType::Float) => spv::Type::float(32),
        ConcreteType::Intrinsic(IntrinsicType::SInt2) => spv::Type::sint(32).of_vector(2),
        ConcreteType::Intrinsic(IntrinsicType::UInt2) => spv::Type::uint(32).of_vector(2),
        ConcreteType::Intrinsic(IntrinsicType::Float2) => spv::Type::float(32).of_vector(2),
        ConcreteType::Intrinsic(IntrinsicType::SInt3) => spv::Type::sint(32).of_vector(3),
        ConcreteType::Intrinsic(IntrinsicType::UInt3) => spv::Type::uint(32).of_vector(3),
        ConcreteType::Intrinsic(IntrinsicType::Float3) => spv::Type::float(32).of_vector(3),
        ConcreteType::Intrinsic(IntrinsicType::SInt4) => spv::Type::sint(32).of_vector(4),
        ConcreteType::Intrinsic(IntrinsicType::UInt4) => spv::Type::uint(32).of_vector(4),
        ConcreteType::Intrinsic(IntrinsicType::Float4) => spv::Type::float(32).of_vector(4),
        ConcreteType::Intrinsic(IntrinsicType::Float2x2) => spv::Type::float(32).of_matrix(2, 2),
        ConcreteType::Intrinsic(IntrinsicType::Float2x3) => spv::Type::float(32).of_matrix(2, 3),
        ConcreteType::Intrinsic(IntrinsicType::Float2x4) => spv::Type::float(32).of_matrix(2, 4),
        ConcreteType::Intrinsic(IntrinsicType::Float3x2) => spv::Type::float(32).of_matrix(3, 2),
        ConcreteType::Intrinsic(IntrinsicType::Float3x3) => spv::Type::float(32).of_matrix(3, 3),
        ConcreteType::Intrinsic(IntrinsicType::Float3x4) => spv::Type::float(32).of_matrix(3, 4),
        ConcreteType::Intrinsic(IntrinsicType::Float4x2) => spv::Type::float(32).of_matrix(4, 2),
        ConcreteType::Intrinsic(IntrinsicType::Float4x3) => spv::Type::float(32).of_matrix(4, 3),
        ConcreteType::Intrinsic(IntrinsicType::Float4x4) => spv::Type::float(32).of_matrix(4, 4),
        ConcreteType::Intrinsic(IntrinsicType::Image2D) => spv::Type::Image {
            sampled_type: spv::ImageComponentType::Scalar(spv::ScalarType::Float(32)),
            dim: spv::asm::Dim::Dim2,
            depth: Some(false),
            arrayed: false,
            multisampled: false,
            sampled: spv::asm::TypeImageSampled::WithReadWriteOps,
            image_format: match symbol_attr.image_format_specifier {
                Some(fmt) => fmt,
                None => spv::asm::ImageFormat::Rgba8,
            },
            access_qualifier: None,
        },
        ConcreteType::Intrinsic(it) => unimplemented!("IntrinsicType: {it:?}"),
        ConcreteType::Generic(_, _) | ConcreteType::GenericVar(_) => {
            unreachable!("pre instantiated type")
        }
        ConcreteType::UnknownIntClass | ConcreteType::UnknownNumberClass => {
            unreachable!("pre instantiated type classes")
        }
        ConcreteType::UserDefined { .. } => unreachable!("pre instantiated user defined type"),
        ConcreteType::Struct(members) => spv::Type::Struct {
            decorations: vec![spv::Decorate::Block],
            member_types: members
                .iter()
                .scan(0, |top, m| {
                    let offset = roundup2(*top, m.ty.std140_alignment().unwrap());
                    *top = offset + m.ty.std140_size().unwrap();

                    Some(spv::TypeStructMember {
                        ty: spv_type(&m.ty, &m.attribute),
                        decorations: vec![spv::Decorate::Offset(offset as _)],
                    })
                })
                .collect(),
        },
        ConcreteType::Tuple(xs) => spv::Type::Struct {
            decorations: Vec::new(),
            member_types: xs
                .iter()
                .map(|x| spv::TypeStructMember {
                    ty: spv_type(x, &SymbolAttribute::default()),
                    decorations: Vec::new(),
                })
                .collect(),
        },
        ConcreteType::Array(t, n) => spv::Type::Array {
            element_type: Box::new(spv_type(&t, &SymbolAttribute::default())),
            length: Box::new(spv::TypeArrayLength::ConstExpr(spv::Constant::from(*n))),
        },
        ConcreteType::Function { .. } => unimplemented!("function type"),
        ConcreteType::IntrinsicTypeConstructor(_) => unreachable!("IntrinsicTypeConstructor"),
        ConcreteType::OverloadedFunctions(_) => unreachable!("Unresolved overloaded functions"),
        ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => {
            unreachable!("Pointer without storage class")
        }
        ConcreteType::Never => unreachable!("never"),
    }
}

pub struct ShaderInterfaceVariableMaps {
    pub builtins: HashMap<spv::asm::Builtin, (SpvSectionLocalId, spv::PointerType)>,
    pub descriptors: HashMap<(u32, u32), (SpvSectionLocalId, spv::PointerType)>,
    pub push_constant: Option<(SpvSectionLocalId, spv::PointerType)>,
    pub push_constant_offset_to_index: HashMap<u32, u32>,
    pub workgroup_shared_memory: HashMap<RefPath, (SpvSectionLocalId, spv::PointerType)>,
}
impl ShaderInterfaceVariableMaps {
    pub fn iter_interface_global_vars(&self) -> impl Iterator<Item = &SpvSectionLocalId> {
        self.builtins
            .values()
            .map(|(v, _)| v)
            .chain(self.descriptors.values().map(|(v, _)| v))
            .chain(self.push_constant.iter().map(|(v, _)| v))
    }
}

pub fn emit_shader_interface_vars<'s>(
    f: &UserDefinedFunctionSymbol,
    fbody: &FunctionBody,
    ctx: &mut SpvModuleEmissionContext,
) -> ShaderInterfaceVariableMaps {
    let mut maps = ShaderInterfaceVariableMaps {
        builtins: HashMap::new(),
        descriptors: HashMap::new(),
        push_constant: None,
        push_constant_offset_to_index: HashMap::new(),
        workgroup_shared_memory: HashMap::new(),
    };
    let mut push_constants = Vec::new();

    fn process(
        path: RefPath,
        attr: &SymbolAttribute,
        ty: &ConcreteType,
        maps: &mut ShaderInterfaceVariableMaps,
        push_constant_entries: &mut Vec<(u32, spv::Type)>,
        ctx: &mut SpvModuleEmissionContext,
    ) {
        let descriptor = match attr {
            &SymbolAttribute {
                descriptor_set_location: Some(set),
                descriptor_set_binding: Some(binding),
                storage_buffer: true,
                ..
            } => Some((set, binding, spv::asm::StorageClass::StorageBuffer)),
            &SymbolAttribute {
                descriptor_set_location: Some(set),
                descriptor_set_binding: Some(binding),
                ..
            } => Some((set, binding, spv::asm::StorageClass::Uniform)),
            _ => None,
        };
        let push_constant = match attr {
            &SymbolAttribute {
                push_constant_offset: Some(offset),
                ..
            } => Some(offset),
            _ => None,
        };
        let builtin = match attr {
            &SymbolAttribute {
                bound_builtin_io: Some(BuiltinInputOutput::VertexID),
                ..
            } => Some(spv::asm::Builtin::VertexIndex),
            &SymbolAttribute {
                bound_builtin_io: Some(BuiltinInputOutput::InstanceID),
                ..
            } => Some(spv::asm::Builtin::InstanceIndex),
            &SymbolAttribute {
                bound_builtin_io: Some(BuiltinInputOutput::GlobalInvocationID),
                ..
            } => Some(spv::asm::Builtin::GlobalInvocationId),
            &SymbolAttribute {
                bound_builtin_io: Some(BuiltinInputOutput::LocalInvocationIndex),
                ..
            } => Some(spv::asm::Builtin::LocalInvocationIndex),
            &SymbolAttribute {
                bound_builtin_io: Some(BuiltinInputOutput::Position),
                ..
            } => Some(spv::asm::Builtin::Position),
            _ => None,
        };
        let workgroup_shared = attr.workgroup_shared;

        match (descriptor, push_constant, builtin, workgroup_shared) {
            (Some((set, binding, sc)), None, None, false) => {
                match maps.descriptors.entry((set, binding)) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let st = spv_type(ty, attr);
                        let gvid = ctx.request_descriptor_set_bound_global_var(
                            set,
                            binding,
                            st.clone(),
                            sc,
                        );
                        e.insert((gvid, st.of_pointer(sc)));
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        assert_eq!(e.get().1, spv_type(ty, attr).of_pointer(sc));
                    }
                }
            }
            (None, Some(o), None, false) => {
                push_constant_entries.push((o, spv_type(ty, attr)));
            }
            (None, None, Some(b), false) => {
                if let std::collections::hash_map::Entry::Vacant(e) = maps.builtins.entry(b) {
                    let st = spv_type(ty, attr);
                    let varid =
                        ctx.declare_global_variable(st.clone(), spv::asm::StorageClass::Input);
                    e.insert((varid, st.of_pointer(spv::asm::StorageClass::Input)));
                }
            }
            (None, None, None, true) => {
                let st = spv_type(ty, attr);
                match maps.workgroup_shared_memory.entry(path) {
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let id = ctx
                            .declare_global_variable(st.clone(), spv::asm::StorageClass::Workgroup);
                        v.insert((id, st.of_pointer(spv::asm::StorageClass::Workgroup)));
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        assert_eq!(e.get().1, st.of_pointer(spv::asm::StorageClass::Workgroup));
                    }
                }
            }
            (None, None, None, false) => match ty {
                ConcreteType::Struct(members) => {
                    for (n, m) in members.iter().enumerate() {
                        process(
                            RefPath::Member(Box::new(path.clone()), n),
                            &m.attribute,
                            &m.ty,
                            maps,
                            push_constant_entries,
                            ctx,
                        );
                    }
                }
                _ => (),
            },
            _ => panic!("Error: conflicting input attribute"),
        }
    }

    for (n, (attr, _, _, ty)) in f.inputs.iter().enumerate() {
        process(
            RefPath::FunctionInput(n),
            attr,
            ty,
            &mut maps,
            &mut push_constants,
            ctx,
        );
    }

    if !push_constants.is_empty() {
        let mut member_types = Vec::new();
        push_constants.sort_by_key(|&(o, _)| o);
        for (o, st) in push_constants {
            match maps.push_constant_offset_to_index.entry(o) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    member_types.push(spv::TypeStructMember {
                        ty: st,
                        decorations: vec![spv::Decorate::Offset(o)],
                    });
                    e.insert(member_types.len() as u32 - 1);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    assert_eq!(&member_types[*e.get() as usize].ty, &st);
                }
            }
        }

        let pc_st = spv::Type::Struct {
            decorations: vec![spv::Decorate::Block],
            member_types,
        };
        let varid = ctx.request_push_constant_global_var(pc_st.clone());
        maps.push_constant = Some((
            varid,
            pc_st.of_pointer(spv::asm::StorageClass::PushConstant),
        ));
    }

    maps
}

pub fn emit_entry_point_spv_ops2<'s>(
    shaderif_variable_maps: &ShaderInterfaceVariableMaps,
    f: &FunctionBody,
    ctx: &mut SpvModuleEmissionContext,
) {
    emit_block(
        shaderif_variable_maps,
        &f.constants,
        &f.blocks,
        BlockRef(0),
        &mut SpvFunctionBodyEmissionContext::new(ctx),
    );
}

pub fn emit_block<'a, 's>(
    shader_if: &ShaderInterfaceVariableMaps,
    consts: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    blocks: &[Block<'a, 's>],
    n: BlockRef,
    ctx: &mut SpvFunctionBodyEmissionContext,
) {
    match blocks[n.0].flow {
        BlockFlowInstruction::Goto(next) => {
            eprintln!("Warn: goto block has no effect");
            emit_block(shader_if, consts, blocks, next, ctx);
        }
        BlockFlowInstruction::StoreRef { ptr, value, after } => {
            let (ptr, ptr_ty) =
                emit_block_instruction(shader_if, consts, &blocks[n.0].instructions, ptr, ctx)
                    .unwrap();
            let (value, value_ty) =
                emit_block_instruction(shader_if, consts, &blocks[n.0].instructions, value, ctx)
                    .unwrap();
            assert_eq!(ptr_ty.clone().dereferenced().unwrap(), value_ty);
            ctx.ops.push(spv::Instruction::Store {
                pointer: ptr,
                object: value,
            });

            emit_block(shader_if, consts, blocks, after.unwrap(), ctx);
        }
        BlockFlowInstruction::IntrinsicImpureFuncall {
            identifier: "Cloth.Intrinsic.ExecutionBarrier",
            ref args,
            after_return,
            ..
        } => {
            assert!(args.is_empty());

            // とりあえずGLSLに倣って固定値で出してるけどこれでいいんだろうか
            let execution_scope = ctx
                .module_ctx
                .request_const_id(spv::Constant::from(spv::asm::Scope::Workgroup as u32));
            let memory_scope = ctx
                .module_ctx
                .request_const_id(spv::Constant::from(spv::asm::Scope::Workgroup as u32));
            let semantics = ctx.module_ctx.request_const_id(spv::Constant::from(
                (spv::asm::MemorySemantics::ACQUIRE_RELEASE
                    | spv::asm::MemorySemantics::WORKGROUP_MEMORY)
                    .bits(),
            ));
            ctx.ops.push(spv::Instruction::ControlBarrier {
                execution: execution_scope,
                memory: memory_scope,
                semantics,
            });

            emit_block(shader_if, consts, blocks, after_return.unwrap(), ctx);
        }
        BlockFlowInstruction::IntrinsicImpureFuncall { identifier, .. } => {
            unimplemented!("Unknown intrinsic: {identifier}")
        }
        BlockFlowInstruction::Funcall {
            callee: RegisterRef(callee),
            ref args,
            result: RegisterRef(result),
            after_return,
        } => {
            unimplemented!("funcall");
            // let (callee, callee_ty) =
            //     emit_block_instruction(shader_if, &block.instructions, callee, ctx).unwrap();
            // let (args, args_ty) = args
            //     .iter()
            //     .map(|RegisterRef(a)| {
            //         emit_block_instruction(shader_if, &block.instructions, a, ctx).unwrap()
            //     })
            //     .collect::<Vec<_>>();
        }
        BlockFlowInstruction::Conditional {
            source,
            r#true,
            r#false,
            merge,
        } => {
            let (source, source_ty) =
                emit_block_instruction(shader_if, consts, &blocks[n.0].instructions, source, ctx)
                    .unwrap();
            assert_eq!(source_ty, spv::Type::Bool);
            // 抜け先を作るのにマージブロックが必要だが、true/falseブロックが具体的にいくつインストラクション数あるかがわからない......
            // IRをもう一個作ってブロックを崩していったほうがよさそう

            let merge_label = ctx.new_id();
            ctx.ops.push(spv::Instruction::Label {
                result: merge_label,
            });
            emit_block(shader_if, consts, blocks, merge, ctx);
        }
        BlockFlowInstruction::ConditionalLoop { .. } => unimplemented!("conditional loop"),
        BlockFlowInstruction::Break => unimplemented!("break"),
        BlockFlowInstruction::Continue => unimplemented!("continue"),
        BlockFlowInstruction::Return(r) => {
            match emit_block_instruction(shader_if, consts, &blocks[n.0].instructions, r, ctx) {
                None => ctx.ops.push(spv::Instruction::Return),
                Some((_, spv::Type::Void)) => ctx.ops.push(spv::Instruction::Return),
                Some(_) => unimplemented!("Return with Value"),
            }
        }
        BlockFlowInstruction::Undetermined => unreachable!("undetermined destination?"),
    }
}

pub fn emit_block_instruction<'a, 's>(
    shader_if: &ShaderInterfaceVariableMaps,
    consts: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    insts: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    n: RegisterRef,
    ctx: &mut SpvFunctionBodyEmissionContext,
) -> Option<(SpvSectionLocalId, spv::Type)> {
    match consts.get(&n).unwrap_or_else(|| &insts[&n]) {
        BlockInstruction::ConstUnit => None,
        BlockInstruction::ConstInt(_) => unreachable!("not reduced const lit"),
        BlockInstruction::ConstNumber(_) => unreachable!("not reduced const lit"),
        BlockInstruction::ConstSInt(ref l) => {
            let value = l.instantiate();
            let const_id = ctx.module_ctx.request_const_id(spv::Constant::from(value));

            Some((const_id, spv::Type::sint(32)))
        }
        BlockInstruction::ConstUInt(ref l) => {
            let value = l.instantiate();
            let const_id = ctx.module_ctx.request_const_id(spv::Constant::from(value));

            Some((const_id, spv::Type::uint(32)))
        }
        BlockInstruction::ConstFloat(ref l) => {
            let value = l.instantiate();
            let const_id = ctx.module_ctx.request_const_id(spv::Constant::from(value));

            Some((const_id, spv::Type::float(32)))
        }
        &BlockInstruction::ImmBool(v) => {
            let const_id = ctx.module_ctx.request_const_id(if v {
                spv::Constant::True {
                    result_type: spv::Type::Bool,
                }
            } else {
                spv::Constant::False {
                    result_type: spv::Type::Bool,
                }
            });

            Some((const_id, spv::Type::Bool))
        }
        BlockInstruction::ImmInt(_) => unreachable!("not reduced const lit"),
        &BlockInstruction::ImmSInt(v) => {
            let const_id = ctx.module_ctx.request_const_id(spv::Constant::from(v));

            Some((const_id, spv::Type::sint(32)))
        }
        &BlockInstruction::ImmUInt(v) => {
            let const_id = ctx.module_ctx.request_const_id(spv::Constant::from(v));

            Some((const_id, spv::Type::uint(32)))
        }
        &BlockInstruction::Cast(src, ref ty) => {
            let (x, xt) = emit_block_instruction(shader_if, consts, insts, src, ctx).unwrap();

            Some(match xt {
                spv::Type::Scalar(spv::ScalarType::Int(32, true)) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::Float) => {
                        ctx.convert_sint_to_float(None, x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Scalar(spv::ScalarType::Int(32, false)) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::Float) => {
                        ctx.convert_uint_to_float(None, x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Scalar(spv::ScalarType::Float(32)) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::SInt) => {
                        ctx.convert_float_to_sint(None, x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::UInt) => {
                        ctx.convert_float_to_uint(None, x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Vector(spv::ScalarType::Int(32, true), c) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::Float2) if c == spv::VectorSize::Two => {
                        ctx.convert_sint_to_float(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::Float3)
                        if c == spv::VectorSize::Three =>
                    {
                        ctx.convert_sint_to_float(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::Float4)
                        if c == spv::VectorSize::Four =>
                    {
                        ctx.convert_sint_to_float(Some(c), x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Vector(spv::ScalarType::Int(32, false), c) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::Float2) if c == spv::VectorSize::Two => {
                        ctx.convert_uint_to_float(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::Float3)
                        if c == spv::VectorSize::Three =>
                    {
                        ctx.convert_uint_to_float(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::Float4)
                        if c == spv::VectorSize::Four =>
                    {
                        ctx.convert_uint_to_float(Some(c), x)
                    }
                    _ => unreachable!(),
                },
                spv::Type::Vector(spv::ScalarType::Float(32), c) => match ty {
                    ConcreteType::Intrinsic(IntrinsicType::SInt2) if c == spv::VectorSize::Two => {
                        ctx.convert_float_to_sint(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::UInt2) if c == spv::VectorSize::Two => {
                        ctx.convert_float_to_uint(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::SInt3)
                        if c == spv::VectorSize::Three =>
                    {
                        ctx.convert_float_to_sint(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::UInt3)
                        if c == spv::VectorSize::Three =>
                    {
                        ctx.convert_float_to_uint(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::SInt4) if c == spv::VectorSize::Four => {
                        ctx.convert_float_to_sint(Some(c), x)
                    }
                    ConcreteType::Intrinsic(IntrinsicType::UInt4) if c == spv::VectorSize::Four => {
                        ctx.convert_float_to_uint(Some(c), x)
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            })
        }
        &BlockInstruction::ConstructIntrinsicComposite(ty, ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|&x| emit_block_instruction(shader_if, consts, insts, x, ctx).unwrap())
                .unzip();

            match ty {
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
                IntrinsicType::Float2 => {
                    assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

                    let rt = spv::Type::float(32).of_vector(2);
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
                                    1 => args.repeat(2),
                                    2 => args,
                                    _ => panic!("Error: component count mismatching"),
                                },
                            });
                    } else {
                        result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::CompositeConstruct {
                            result_type,
                            result: result_id,
                            constituents: match args.len() {
                                1 => args.repeat(2),
                                2 => args,
                                _ => panic!("Error: component count mismatching"),
                            },
                        });
                    };

                    Some((result_id, rt))
                }
                _ => unimplemented!("Composite construction for {ty:?}"),
            }
        }
        BlockInstruction::InstantiateIntrinsicTypeClass(_, _) => {
            unreachable!("not instantiated intrinsic type?")
        }
        BlockInstruction::PromoteIntToNumber(_) => unreachable!("not processed promotion"),
        BlockInstruction::ConstructStruct(_) => unimplemented!("ConstructStruct"),
        BlockInstruction::ConstructTuple(_) => unimplemented!("ConstructTuple"),
        BlockInstruction::BuiltinIORef(b) => {
            let (id, ty) = shader_if.builtins[match b {
                BuiltinInputOutput::VertexID => &spv::asm::Builtin::VertexIndex,
                BuiltinInputOutput::InstanceID => &spv::asm::Builtin::InstanceIndex,
                BuiltinInputOutput::Position => &spv::asm::Builtin::Position,
                BuiltinInputOutput::LocalInvocationIndex => {
                    &spv::asm::Builtin::LocalInvocationIndex
                }
                BuiltinInputOutput::GlobalInvocationID => &spv::asm::Builtin::GlobalInvocationId,
            }]
            .clone();

            Some((id, ty.into()))
        }
        &BlockInstruction::DescriptorRef { set, binding } => {
            let (id, ty) = shader_if.descriptors[&(set, binding)].clone();

            Some((id, ty.into()))
        }
        &BlockInstruction::PushConstantRef(offset) => {
            let (vid, vty) = shader_if.push_constant.clone().unwrap();
            let mindex = shader_if.push_constant_offset_to_index[&offset];
            let result_ty = match vty.base {
                spv::Type::Struct { member_types, .. } => member_types[mindex as usize]
                    .ty
                    .clone()
                    .of_pointer(vty.storage_class),
                _ => unreachable!(),
            };

            let mindex_const = ctx.module_ctx.request_const_id(spv::Constant::from(mindex));
            Some((
                ctx.access_chain(result_ty.clone().into(), vid, vec![mindex_const]),
                result_ty.into(),
            ))
        }
        BlockInstruction::ScopeLocalVarRef(_, _) => unimplemented!("ScopeLocalVarRef"),
        BlockInstruction::FunctionInputVarRef(_, _) => unimplemented!("FunctionInputVarRef"),
        BlockInstruction::UserDefinedFunctionRef(_, _) => unreachable!("not inlined function"),
        BlockInstruction::IntrinsicFunctionRef(_) => {
            unreachable!("unresolved intrinsic function ref")
        }
        BlockInstruction::IntrinsicTypeConstructorRef(_) => {
            unreachable!("unresolved intrinsic type constructor ref")
        }
        BlockInstruction::TupleRef(_, _) => unimplemented!("TupleRef"),
        BlockInstruction::MemberRef(_, _) => unreachable!("MemberRef"),
        BlockInstruction::WorkgroupSharedMemoryRef(path) => {
            let (vid, vty) = shader_if.workgroup_shared_memory[&path].clone();

            Some((vid, vty.into()))
        }
        BlockInstruction::StaticPathRef(_) => unimplemented!("StaticPathRef"),
        &BlockInstruction::ArrayRef { source, index } => {
            let (source, source_ty) =
                emit_block_instruction(shader_if, consts, insts, source, ctx).unwrap();
            let (index, _index_ty) =
                emit_block_instruction(shader_if, consts, insts, index, ctx).unwrap();
            let element_ptr_ty = match source_ty {
                spv::Type::Pointer(ptr) => match ptr.base {
                    spv::Type::Array { element_type, .. } => {
                        element_type.of_pointer(ptr.storage_class)
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            Some((
                ctx.access_chain(element_ptr_ty.clone().into(), source, vec![index]),
                element_ptr_ty.into(),
            ))
        }
        BlockInstruction::SwizzleRef(_, _) => {
            unimplemented!("SwizzleRef")
        }
        &BlockInstruction::Swizzle(src, ref elements) => {
            let (x, xt) = emit_block_instruction(shader_if, consts, insts, src, ctx).unwrap();
            let spv::Type::Vector(component_type, _) = xt else {
                unreachable!();
            };

            match &elements[..] {
                &[a] => Some((
                    ctx.composite_extract(component_type.clone(), x, vec![a as _]),
                    component_type.into(),
                )),
                &[a, b] => {
                    let rt = component_type.of_vector(spv::VectorSize::Two);
                    Some((
                        ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _]),
                        rt.into(),
                    ))
                }
                &[a, b, c] => {
                    let rt = component_type.of_vector(spv::VectorSize::Three);
                    Some((
                        ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _]),
                        rt.into(),
                    ))
                }
                &[a, b, c, d] => {
                    let rt = component_type.of_vector(spv::VectorSize::Four);
                    Some((
                        ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _, d as _]),
                        rt.into(),
                    ))
                }
                _ => unreachable!(),
            }
        }
        &BlockInstruction::LoadRef(ptr) => {
            let (ptr, ptr_ty) = emit_block_instruction(shader_if, consts, insts, ptr, ctx).unwrap();
            let value_ty = ptr_ty.dereferenced().unwrap();

            let result = ctx.load(value_ty.clone(), ptr);
            Some((result, value_ty))
        }
        &BlockInstruction::IntrinsicBinaryOp(left, op, right) => {
            let (l, lt) = emit_block_instruction(shader_if, consts, insts, left, ctx).unwrap();
            let (r, rt) = emit_block_instruction(shader_if, consts, insts, right, ctx).unwrap();
            assert_eq!(lt, rt);

            match op {
                IntrinsicBinaryOperation::Add => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some((
                        match sv.scalar() {
                            spv::ScalarType::Int(_, _) => ctx.iadd(lt.clone(), l, r),
                            spv::ScalarType::Float(_) => ctx.fadd(lt.clone(), l, r),
                            _ => unreachable!(),
                        },
                        lt,
                    ))
                }
                IntrinsicBinaryOperation::Sub => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some((
                        match sv.scalar() {
                            spv::ScalarType::Int(_, _) => ctx.isub(lt.clone(), l, r),
                            spv::ScalarType::Float(_) => ctx.fsub(lt.clone(), l, r),
                            _ => unreachable!(),
                        },
                        lt,
                    ))
                }
                IntrinsicBinaryOperation::Mul => match (lt, rt) {
                    (
                        spv::Type::Vector(spv::ScalarType::Float(w1), component_count),
                        spv::Type::Scalar(spv::ScalarType::Float(w2)),
                    ) if w1 == w2 => {
                        let rt: spv::Type =
                            spv::ScalarType::Float(w1).of_vector(component_count).into();
                        Some((ctx.vector_times_scalar(rt.clone(), l, r), rt))
                    }
                    (
                        spv::Type::Matrix(
                            spv::VectorType(spv::ScalarType::Float(w1), row_count),
                            column_count,
                        ),
                        spv::Type::Scalar(spv::ScalarType::Float(w2)),
                    ) if w1 == w2 => {
                        let rt: spv::Type = spv::ScalarType::Float(w1)
                            .of_matrix(row_count, column_count as _)
                            .into();
                        Some((ctx.matrix_times_scalar(rt.clone(), l, r), rt))
                    }
                    (
                        spv::Type::Matrix(
                            spv::VectorType(spv::ScalarType::Float(w1), component_count),
                            ccl,
                        ),
                        spv::Type::Vector(spv::ScalarType::Float(w2), ccr),
                    ) if ccl.count() == ccr.count() && w1 == w2 => {
                        let rt: spv::Type =
                            spv::ScalarType::Float(w1).of_vector(component_count).into();
                        Some((ctx.matrix_times_vector(rt.clone(), l, r), rt))
                    }
                    (
                        spv::Type::Vector(spv::ScalarType::Float(w1), ccl),
                        spv::Type::Matrix(
                            spv::VectorType(spv::ScalarType::Float(w2), component_count),
                            ccr,
                        ),
                    ) if ccl == component_count && w1 == w2 => {
                        let rt: spv::Type = spv::ScalarType::Float(w2).of_vector(ccr.into()).into();
                        Some((ctx.vector_times_matrix(rt.clone(), l, r), rt))
                    }
                    (
                        spv::Type::Matrix(spv::VectorType(spv::ScalarType::Float(w1), rl), cl),
                        spv::Type::Matrix(spv::VectorType(spv::ScalarType::Float(w2), rr), cr),
                    ) if w1 == w2 && cl.count() == rr.count() => {
                        let rt: spv::Type = spv::ScalarType::Float(w1).of_matrix(rl, cr).into();
                        Some((ctx.matrix_times_matrix(rt.clone(), l, r), rt))
                    }
                    (a, b) if a == b => Some((
                        match a {
                            spv::Type::Scalar(spv::ScalarType::Int(_, _))
                            | spv::Type::Vector(spv::ScalarType::Int(_, _), _) => {
                                ctx.imul(a.clone(), l, r)
                            }
                            spv::Type::Scalar(spv::ScalarType::Float(_))
                            | spv::Type::Vector(spv::ScalarType::Float(_), _) => {
                                ctx.fmul(a.clone(), l, r)
                            }
                            _ => unreachable!(),
                        },
                        a,
                    )),
                    _ => unreachable!(),
                },
                IntrinsicBinaryOperation::Div => Some((
                    match lt {
                        spv::Type::Scalar(spv::ScalarType::Int(_, true))
                        | spv::Type::Vector(spv::ScalarType::Int(_, true), _) => {
                            ctx.sdiv(lt.clone(), l, r)
                        }
                        spv::Type::Scalar(spv::ScalarType::Int(_, false))
                        | spv::Type::Vector(spv::ScalarType::Int(_, false), _) => {
                            ctx.udiv(lt.clone(), l, r)
                        }
                        spv::Type::Scalar(spv::ScalarType::Float(_))
                        | spv::Type::Vector(spv::ScalarType::Float(_), _) => {
                            ctx.fdiv(lt.clone(), l, r)
                        }
                        _ => unreachable!(),
                    },
                    lt,
                )),
                IntrinsicBinaryOperation::Rem => Some((
                    match lt {
                        spv::Type::Scalar(spv::ScalarType::Int(_, true))
                        | spv::Type::Vector(spv::ScalarType::Int(_, true), _) => {
                            ctx.srem(lt.clone(), l, r)
                        }
                        spv::Type::Scalar(spv::ScalarType::Int(_, false))
                        | spv::Type::Vector(spv::ScalarType::Int(_, false), _) => {
                            ctx.umod(lt.clone(), l, r)
                        }
                        spv::Type::Scalar(spv::ScalarType::Float(_))
                        | spv::Type::Vector(spv::ScalarType::Float(_), _) => {
                            ctx.frem(lt.clone(), l, r)
                        }
                        _ => unreachable!(),
                    },
                    lt,
                )),
                IntrinsicBinaryOperation::Pow => {
                    let rt: spv::Type = match lt {
                        spv::Type::Scalar(spv::ScalarType::Float(width)) => {
                            spv::ScalarType::Float(width).into()
                        }
                        spv::Type::Vector(spv::ScalarType::Float(width), component_count) => {
                            spv::ScalarType::Float(width)
                                .of_vector(component_count)
                                .into()
                        }
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
                IntrinsicBinaryOperation::BitAnd => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };
                    assert!(matches!(sv.scalar(), &spv::ScalarType::Int(_, _)));

                    let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                    ctx.ops.push(spv::Instruction::BitwiseAnd {
                        result_type,
                        result,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result, lt))
                }
                IntrinsicBinaryOperation::BitOr => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };
                    assert!(matches!(sv.scalar(), spv::ScalarType::Int(_, _)));

                    let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                    ctx.ops.push(spv::Instruction::BitwiseOr {
                        result_type,
                        result,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result, lt))
                }
                IntrinsicBinaryOperation::BitXor => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };
                    assert!(matches!(sv.scalar(), spv::ScalarType::Int(_, _)));

                    let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                    ctx.ops.push(spv::Instruction::BitwiseXor {
                        result_type,
                        result,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result, lt))
                }
                IntrinsicBinaryOperation::LeftShift => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    match sv.scalar() {
                        spv::ScalarType::Int(_, _) => {
                            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                            ctx.ops.push(spv::Instruction::ShiftLeftLogical {
                                result_type,
                                result,
                                base: l,
                                shift: r,
                            });
                            Some((result, lt))
                        }
                        _ => unreachable!(),
                    }
                }
                IntrinsicBinaryOperation::RightShift => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    match sv.scalar() {
                        spv::ScalarType::Int(_, true) => {
                            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                            ctx.ops.push(spv::Instruction::ShiftRightArithmetic {
                                result_type,
                                result,
                                base: l,
                                shift: r,
                            });
                            Some((result, lt))
                        }
                        spv::ScalarType::Int(_, false) => {
                            let (result_type, result) = ctx.issue_typed_ids(lt.clone());
                            ctx.ops.push(spv::Instruction::ShiftRightLogical {
                                result_type,
                                result,
                                base: l,
                                shift: r,
                            });
                            Some((result, lt))
                        }
                        _ => unreachable!(),
                    }
                }
                IntrinsicBinaryOperation::Eq => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.equal(
                        EqCompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::Ne => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.not_equal(
                        EqCompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::Lt => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.less_than(
                        CompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::Le => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.less_than_eq(
                        CompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::Gt => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.greater_than(
                        CompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::Ge => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };

                    Some(ctx.greater_than_eq(
                        CompareOperandClass::of(sv.scalar()),
                        sv.vector_size(),
                        l,
                        r,
                    ))
                }
                IntrinsicBinaryOperation::LogAnd => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };
                    assert_eq!(sv.scalar(), &spv::ScalarType::Bool);

                    Some(ctx.log_and(sv.vector_size(), l, r))
                }
                IntrinsicBinaryOperation::LogOr => {
                    let Some(sv) = lt.scalar_or_vector_view() else {
                        unreachable!();
                    };
                    assert_eq!(sv.scalar(), &spv::ScalarType::Bool);

                    Some(ctx.log_or(sv.vector_size(), l, r))
                }
            }
        }
        &BlockInstruction::IntrinsicUnaryOp(src, op) => {
            let (x, xt) = emit_block_instruction(shader_if, consts, insts, src, ctx).unwrap();
            let Some(xtsv) = xt.clone().scalar_or_vector() else {
                unreachable!();
            };

            match op {
                IntrinsicUnaryOperation::Neg => Some((
                    match xtsv.scalar() {
                        spv::ScalarType::Int(_, true) => ctx.snegate(xt.clone(), x),
                        spv::ScalarType::Float(_) => ctx.fnegate(xt.clone(), x),
                        _ => unreachable!(),
                    },
                    xt,
                )),
                IntrinsicUnaryOperation::BitNot => unimplemented!("Unary:BitNot"),
                IntrinsicUnaryOperation::LogNot => unimplemented!("Unary:LogNot"),
            }
        }
        BlockInstruction::Phi(_) => unimplemented!("Phi"),
        BlockInstruction::RegisterAlias(_) => unreachable!("unresolved register alias"),
        BlockInstruction::PureFuncall(_, _) => unreachable!("all functions must be inlined"),
        &BlockInstruction::PureIntrinsicCall("Cloth.Intrinsic.SubpassLoad", ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_block_instruction(shader_if, consts, insts, *x, ctx).unwrap())
                .unzip();
            assert!(arg_ty.len() == 1 && arg_ty[0] == spv::Type::SUBPASS_DATA_IMAGE_TYPE);

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
        &BlockInstruction::PureIntrinsicCall("Cloth.Intrinsic.Dot#Float3", ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_block_instruction(shader_if, consts, insts, *x, ctx).unwrap())
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
        &BlockInstruction::PureIntrinsicCall("Cloth.Intrinsic.Normalize#Float3", ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_block_instruction(shader_if, consts, insts, *x, ctx).unwrap())
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
        &BlockInstruction::PureIntrinsicCall("Cloth.Intrinsic.Transpose#Float4x4", ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_block_instruction(shader_if, consts, insts, *x, ctx).unwrap())
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
        &BlockInstruction::PureIntrinsicCall("Cloth.Intrinsic.SampleAt#Texture2D", ref args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_block_instruction(shader_if, consts, insts, *x, ctx).unwrap())
                .unzip();
            assert_eq!(arg_ty.len(), 2);
            assert!(matches!(arg_ty[0], spv::Type::SampledImage { .. }));
            assert_eq!(arg_ty[1], spv::Type::float(32).of_vector(2));

            let rt = spv::Type::float(32).of_vector(4);
            let (result_type, result) = ctx.issue_typed_ids(rt.clone());
            ctx.ops.push(spv::Instruction::ImageSampleImplicitLod {
                result_type,
                result,
                sampled_image: args[0],
                coordinate: args[1],
            });
            Some((result, rt))
        }
        BlockInstruction::PureIntrinsicCall(id, ref args) => panic!(
            "Error: unknown intrinsic call {id}({})",
            args.iter()
                .map(|r| format!("r{}", r.0))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
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

    for a in ep.global_variables.outputs.iter() {
        let gvid = ctx.declare_global_variable(a.ty.clone(), spv::asm::StorageClass::Output);
        ctx.decorate(gvid, &a.decorations);

        entry_point_maps.output_global_vars.push(gvid);
        entry_point_maps.interface_global_vars.push(gvid);
    }

    for a in ep.global_variables.inputs.iter() {
        let vid = ctx.declare_global_variable(a.ty.clone(), spv::asm::StorageClass::Input);
        ctx.decorate(vid, &a.decorations);

        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(vid, a.ty.clone(), spv::asm::StorageClass::Input),
        );
        entry_point_maps.interface_global_vars.push(vid);
    }

    for a in ep.global_variables.uniforms.iter() {
        let storage_class = match a.ty {
            spv::Type::Image { .. } | spv::Type::SampledImage { .. } => {
                spv::asm::StorageClass::UniformConstant
            }
            _ => spv::asm::StorageClass::Uniform,
        };
        let vid = ctx.declare_global_variable(a.ty.clone(), storage_class);
        ctx.decorate(vid, &a.decorations);

        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(vid, a.ty.clone(), storage_class),
        );
        entry_point_maps.interface_global_vars.push(vid);
    }

    if !ep.global_variables.push_constants.is_empty() {
        let block_ty = spv::Type::Struct {
            decorations: vec![spv::Decorate::Block],
            member_types: ep
                .global_variables
                .push_constants
                .iter()
                .map(|x| {
                    let mut decorations = vec![spv::Decorate::Offset(x.offset)];
                    if let spv::Type::Matrix(ref r, _) = x.ty {
                        decorations.extend([
                            spv::Decorate::ColMajor,
                            spv::Decorate::MatrixStride(r.matrix_stride().unwrap()),
                        ]);
                    }

                    spv::TypeStructMember {
                        ty: x.ty.clone(),
                        decorations,
                    }
                })
                .collect(),
        };
        let block_var = ctx.declare_global_variable(block_ty, spv::asm::StorageClass::PushConstant);
        for (n, a) in ep.global_variables.push_constants.iter().enumerate() {
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

    for v in ep.global_variables.workgroup_shared_vars.iter() {
        let storage_class = spv::asm::StorageClass::Workgroup;
        let vid = ctx.declare_global_variable(v.ty.clone(), storage_class);

        entry_point_maps.refpath_to_global_var.insert(
            v.original_refpath.clone(),
            GlobalAccessType::Direct(vid, v.ty.clone(), storage_class),
        );
    }

    entry_point_maps
}

// pub fn emit_function_body_spv_ops(
//     body: &[(SimplifiedExpression, ConcreteType)],
//     ctx: &mut SpvFunctionBodyEmissionContext,
// ) {
//     for (n, (_, _)) in body.iter().enumerate() {
//         if !ExprRef(n).is_pure(body) {
//             emit_expr_spv_ops(body, n, ctx);
//         }
//     }
// }

// fn emit_expr_spv_ops(
//     body: &[(SimplifiedExpression, ConcreteType)],
//     expr_id: usize,
//     ctx: &mut SpvFunctionBodyEmissionContext,
// ) -> Option<(SpvSectionLocalId, spv::Type)> {
//     if let Some(r) = ctx.emitted_expression_id.get(&expr_id) {
//         return r.clone();
//     }

//     let result = match &body[expr_id].0 {
//         SimplifiedExpression::Select(c, t, e) => {
//             let (c, ct) = emit_expr_spv_ops(body, c.0, ctx).unwrap();

//             let requires_control_flow_branching = ct != spv::Type::Scalar(spv::ScalarType::Bool)
//                 || matches!(
//                     (&body[t.0].0, &body[e.0].0),
//                     (SimplifiedExpression::ScopedBlock { .. }, _)
//                         | (_, SimplifiedExpression::ScopedBlock { .. })
//                 );

//             if requires_control_flow_branching {
//                 // impure select
//                 unimplemented!("Control Flow Branching strategy");
//             } else {
//                 let (t, tt) = emit_expr_spv_ops(body, t.0, ctx).unwrap();
//                 let (e, et) = emit_expr_spv_ops(body, e.0, ctx).unwrap();
//                 assert_eq!(tt, et);

//                 Some(ctx.select(tt.clone(), c, t, e))
//             }
//         }
//         SimplifiedExpression::ConstructIntrinsicComposite(it, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|&x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();

//             match it {
//                 IntrinsicType::Float4 => {
//                     assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

//                     let rt = spv::Type::float(32).of_vector(4);
//                     let result_type = ctx.module_ctx.request_type_id(rt.clone());
//                     let is_constant = args
//                         .iter()
//                         .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

//                     let result_id;
//                     if is_constant {
//                         result_id = ctx.module_ctx.new_type_const_id();
//                         ctx.module_ctx
//                             .type_const_ops
//                             .push(spv::Instruction::ConstantComposite {
//                                 result_type,
//                                 result: result_id,
//                                 constituents: match args.len() {
//                                     1 => args.repeat(4),
//                                     2 => args.repeat(2),
//                                     4 => args,
//                                     _ => panic!("Error: component count mismatching"),
//                                 },
//                             });
//                     } else {
//                         result_id = ctx.new_id();
//                         ctx.ops.push(spv::Instruction::CompositeConstruct {
//                             result_type,
//                             result: result_id,
//                             constituents: match args.len() {
//                                 1 => args.repeat(4),
//                                 2 => args.repeat(2),
//                                 4 => args,
//                                 _ => panic!("Error: component count mismatching"),
//                             },
//                         });
//                     };

//                     Some((result_id, rt))
//                 }
//                 IntrinsicType::Float3 => {
//                     assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

//                     let rt = spv::Type::float(32).of_vector(3);
//                     let result_type = ctx.module_ctx.request_type_id(rt.clone());
//                     let is_constant = args
//                         .iter()
//                         .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

//                     let result_id;
//                     if is_constant {
//                         result_id = ctx.module_ctx.new_type_const_id();
//                         ctx.module_ctx
//                             .type_const_ops
//                             .push(spv::Instruction::ConstantComposite {
//                                 result_type,
//                                 result: result_id,
//                                 constituents: match args.len() {
//                                     1 => args.repeat(3),
//                                     3 => args,
//                                     _ => panic!("Error: component count mismatching"),
//                                 },
//                             });
//                     } else {
//                         result_id = ctx.new_id();
//                         ctx.ops.push(spv::Instruction::CompositeConstruct {
//                             result_type,
//                             result: result_id,
//                             constituents: match args.len() {
//                                 1 => args.repeat(3),
//                                 3 => args,
//                                 _ => panic!("Error: component count mismatching"),
//                             },
//                         });
//                     };

//                     Some((result_id, rt))
//                 }
//                 IntrinsicType::Float2 => {
//                     assert!(arg_ty.iter().all(|x| *x == spv::Type::float(32)));

//                     let rt = spv::Type::float(32).of_vector(2);
//                     let result_type = ctx.module_ctx.request_type_id(rt.clone());
//                     let is_constant = args
//                         .iter()
//                         .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

//                     let result_id;
//                     if is_constant {
//                         result_id = ctx.module_ctx.new_type_const_id();
//                         ctx.module_ctx
//                             .type_const_ops
//                             .push(spv::Instruction::ConstantComposite {
//                                 result_type,
//                                 result: result_id,
//                                 constituents: match args.len() {
//                                     1 => args.repeat(2),
//                                     2 => args,
//                                     _ => panic!("Error: component count mismatching"),
//                                 },
//                             });
//                     } else {
//                         result_id = ctx.new_id();
//                         ctx.ops.push(spv::Instruction::CompositeConstruct {
//                             result_type,
//                             result: result_id,
//                             constituents: match args.len() {
//                                 1 => args.repeat(2),
//                                 2 => args,
//                                 _ => panic!("Error: component count mismatching"),
//                             },
//                         });
//                     };

//                     Some((result_id, rt))
//                 }
//                 _ => unimplemented!("Composite construction for {it:?}"),
//             }
//         }
//         SimplifiedExpression::LogAnd(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };
//             assert_eq!(sv.scalar(), &spv::ScalarType::Bool);

//             Some(ctx.log_and(sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::LogOr(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };
//             assert_eq!(sv.scalar(), &spv::ScalarType::Bool);

//             Some(ctx.log_or(sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::Eq(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.equal(
//                 EqCompareOperandClass::of(sv.scalar()),
//                 sv.vector_size(),
//                 l,
//                 r,
//             ))
//         }
//         SimplifiedExpression::Ne(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.not_equal(
//                 EqCompareOperandClass::of(sv.scalar()),
//                 sv.vector_size(),
//                 l,
//                 r,
//             ))
//         }
//         SimplifiedExpression::Lt(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.less_than(CompareOperandClass::of(sv.scalar()), sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::Le(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.less_than_eq(CompareOperandClass::of(sv.scalar()), sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::Gt(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.greater_than(CompareOperandClass::of(sv.scalar()), sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::Ge(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some(ctx.greater_than_eq(CompareOperandClass::of(sv.scalar()), sv.vector_size(), l, r))
//         }
//         SimplifiedExpression::BitAnd(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };
//             assert!(matches!(sv.scalar(), &spv::ScalarType::Int(_, _)));

//             let (result_type, result) = ctx.issue_typed_ids(lt.clone());
//             ctx.ops.push(spv::Instruction::BitwiseAnd {
//                 result_type,
//                 result,
//                 operand1: l,
//                 operand2: r,
//             });
//             Some((result, lt))
//         }
//         SimplifiedExpression::BitOr(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };
//             assert!(matches!(sv.scalar(), spv::ScalarType::Int(_, _)));

//             let (result_type, result) = ctx.issue_typed_ids(lt.clone());
//             ctx.ops.push(spv::Instruction::BitwiseOr {
//                 result_type,
//                 result,
//                 operand1: l,
//                 operand2: r,
//             });
//             Some((result, lt))
//         }
//         SimplifiedExpression::BitXor(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };
//             assert!(matches!(sv.scalar(), spv::ScalarType::Int(_, _)));

//             let (result_type, result) = ctx.issue_typed_ids(lt.clone());
//             ctx.ops.push(spv::Instruction::BitwiseXor {
//                 result_type,
//                 result,
//                 operand1: l,
//                 operand2: r,
//             });
//             Some((result, lt))
//         }
//         SimplifiedExpression::Add(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some((
//                 match sv.scalar() {
//                     spv::ScalarType::Int(_, _) => ctx.iadd(lt.clone(), l, r),
//                     spv::ScalarType::Float(_) => ctx.fadd(lt.clone(), l, r),
//                     _ => unreachable!(),
//                 },
//                 lt,
//             ))
//         }
//         SimplifiedExpression::Sub(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);
//             let Some(sv) = lt.scalar_or_vector_view() else {
//                 unreachable!();
//             };

//             Some((
//                 match sv.scalar() {
//                     spv::ScalarType::Int(_, _) => ctx.isub(lt.clone(), l, r),
//                     spv::ScalarType::Float(_) => ctx.fsub(lt.clone(), l, r),
//                     _ => unreachable!(),
//                 },
//                 lt,
//             ))
//         }
//         SimplifiedExpression::Mul(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();

//             match (lt, rt) {
//                 (
//                     spv::Type::Vector(spv::ScalarType::Float(w1), component_count),
//                     spv::Type::Scalar(spv::ScalarType::Float(w2)),
//                 ) if w1 == w2 => {
//                     let rt: spv::Type =
//                         spv::ScalarType::Float(w1).of_vector(component_count).into();
//                     Some((ctx.vector_times_scalar(rt.clone(), l, r), rt))
//                 }
//                 (
//                     spv::Type::Matrix(
//                         spv::VectorType(spv::ScalarType::Float(w1), row_count),
//                         column_count,
//                     ),
//                     spv::Type::Scalar(spv::ScalarType::Float(w2)),
//                 ) if w1 == w2 => {
//                     let rt: spv::Type = spv::ScalarType::Float(w1)
//                         .of_matrix(row_count, column_count as _)
//                         .into();
//                     Some((ctx.matrix_times_scalar(rt.clone(), l, r), rt))
//                 }
//                 (
//                     spv::Type::Matrix(
//                         spv::VectorType(spv::ScalarType::Float(w1), component_count),
//                         ccl,
//                     ),
//                     spv::Type::Vector(spv::ScalarType::Float(w2), ccr),
//                 ) if ccl.count() == ccr.count() && w1 == w2 => {
//                     let rt: spv::Type =
//                         spv::ScalarType::Float(w1).of_vector(component_count).into();
//                     Some((ctx.matrix_times_vector(rt.clone(), l, r), rt))
//                 }
//                 (
//                     spv::Type::Vector(spv::ScalarType::Float(w1), ccl),
//                     spv::Type::Matrix(
//                         spv::VectorType(spv::ScalarType::Float(w2), component_count),
//                         ccr,
//                     ),
//                 ) if ccl == component_count && w1 == w2 => {
//                     let rt: spv::Type = spv::ScalarType::Float(w2).of_vector(ccr.into()).into();
//                     Some((ctx.vector_times_matrix(rt.clone(), l, r), rt))
//                 }
//                 (
//                     spv::Type::Matrix(spv::VectorType(spv::ScalarType::Float(w1), rl), cl),
//                     spv::Type::Matrix(spv::VectorType(spv::ScalarType::Float(w2), rr), cr),
//                 ) if w1 == w2 && cl.count() == rr.count() => {
//                     let rt: spv::Type = spv::ScalarType::Float(w1).of_matrix(rl, cr).into();
//                     Some((ctx.matrix_times_matrix(rt.clone(), l, r), rt))
//                 }
//                 (a, b) if a == b => Some((
//                     match a {
//                         spv::Type::Scalar(spv::ScalarType::Int(_, _))
//                         | spv::Type::Vector(spv::ScalarType::Int(_, _), _) => {
//                             ctx.imul(a.clone(), l, r)
//                         }
//                         spv::Type::Scalar(spv::ScalarType::Float(_))
//                         | spv::Type::Vector(spv::ScalarType::Float(_), _) => {
//                             ctx.fmul(a.clone(), l, r)
//                         }
//                         _ => unreachable!(),
//                     },
//                     a,
//                 )),
//                 _ => unreachable!(),
//             }
//         }
//         SimplifiedExpression::Div(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);

//             Some((
//                 match lt {
//                     spv::Type::Scalar(spv::ScalarType::Int(_, true))
//                     | spv::Type::Vector(spv::ScalarType::Int(_, true), _) => {
//                         ctx.sdiv(lt.clone(), l, r)
//                     }
//                     spv::Type::Scalar(spv::ScalarType::Int(_, false))
//                     | spv::Type::Vector(spv::ScalarType::Int(_, false), _) => {
//                         ctx.udiv(lt.clone(), l, r)
//                     }
//                     spv::Type::Scalar(spv::ScalarType::Float(_))
//                     | spv::Type::Vector(spv::ScalarType::Float(_), _) => ctx.fdiv(lt.clone(), l, r),
//                     _ => unreachable!(),
//                 },
//                 lt,
//             ))
//         }
//         SimplifiedExpression::Rem(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);

//             Some((
//                 match lt {
//                     spv::Type::Scalar(spv::ScalarType::Int(_, true))
//                     | spv::Type::Vector(spv::ScalarType::Int(_, true), _) => {
//                         ctx.srem(lt.clone(), l, r)
//                     }
//                     spv::Type::Scalar(spv::ScalarType::Int(_, false))
//                     | spv::Type::Vector(spv::ScalarType::Int(_, false), _) => {
//                         ctx.umod(lt.clone(), l, r)
//                     }
//                     spv::Type::Scalar(spv::ScalarType::Float(_))
//                     | spv::Type::Vector(spv::ScalarType::Float(_), _) => ctx.frem(lt.clone(), l, r),
//                     _ => unreachable!(),
//                 },
//                 lt,
//             ))
//         }
//         SimplifiedExpression::Pow(l, r) => {
//             let (l, lt) = emit_expr_spv_ops(body, l.0, ctx).unwrap();
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             assert_eq!(lt, rt);

//             let rt: spv::Type = match lt {
//                 spv::Type::Scalar(spv::ScalarType::Float(width)) => {
//                     spv::ScalarType::Float(width).into()
//                 }
//                 spv::Type::Vector(spv::ScalarType::Float(width), component_count) => {
//                     spv::ScalarType::Float(width)
//                         .of_vector(component_count)
//                         .into()
//                 }
//                 _ => unreachable!(),
//             };

//             let glsl_std_450_lib_set_id =
//                 ctx.module_ctx.request_ext_inst_set("GLSL.std.450".into());
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::ExtInst {
//                 result_type,
//                 result,
//                 set: glsl_std_450_lib_set_id,
//                 instruction: 26,
//                 operands: vec![l, r],
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::Neg(x) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             let Some(xtsv) = xt.clone().scalar_or_vector() else {
//                 unreachable!();
//             };

//             Some((
//                 match xtsv.scalar() {
//                     spv::ScalarType::Int(_, true) => ctx.snegate(xt.clone(), x),
//                     spv::ScalarType::Float(_) => ctx.fnegate(xt.clone(), x),
//                     _ => unreachable!(),
//                 },
//                 xt,
//             ))
//         }
//         SimplifiedExpression::Cast(x, t) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();

//             Some(match xt {
//                 spv::Type::Scalar(spv::ScalarType::Int(32, true)) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::Float) => {
//                         ctx.convert_sint_to_float(None, x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 spv::Type::Scalar(spv::ScalarType::Int(32, false)) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::Float) => {
//                         ctx.convert_uint_to_float(None, x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 spv::Type::Scalar(spv::ScalarType::Float(32)) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::SInt) => {
//                         ctx.convert_float_to_sint(None, x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::UInt) => {
//                         ctx.convert_float_to_uint(None, x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 spv::Type::Vector(spv::ScalarType::Int(32, true), c) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::Float2) if c == spv::VectorSize::Two => {
//                         ctx.convert_sint_to_float(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::Float3)
//                         if c == spv::VectorSize::Three =>
//                     {
//                         ctx.convert_sint_to_float(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::Float4)
//                         if c == spv::VectorSize::Four =>
//                     {
//                         ctx.convert_sint_to_float(Some(c), x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 spv::Type::Vector(spv::ScalarType::Int(32, false), c) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::Float2) if c == spv::VectorSize::Two => {
//                         ctx.convert_uint_to_float(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::Float3)
//                         if c == spv::VectorSize::Three =>
//                     {
//                         ctx.convert_uint_to_float(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::Float4)
//                         if c == spv::VectorSize::Four =>
//                     {
//                         ctx.convert_uint_to_float(Some(c), x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 spv::Type::Vector(spv::ScalarType::Float(32), c) => match t {
//                     ConcreteType::Intrinsic(IntrinsicType::SInt2) if c == spv::VectorSize::Two => {
//                         ctx.convert_float_to_sint(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::UInt2) if c == spv::VectorSize::Two => {
//                         ctx.convert_float_to_uint(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::SInt3)
//                         if c == spv::VectorSize::Three =>
//                     {
//                         ctx.convert_float_to_sint(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::UInt3)
//                         if c == spv::VectorSize::Three =>
//                     {
//                         ctx.convert_float_to_uint(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::SInt4) if c == spv::VectorSize::Four => {
//                         ctx.convert_float_to_sint(Some(c), x)
//                     }
//                     ConcreteType::Intrinsic(IntrinsicType::UInt4) if c == spv::VectorSize::Four => {
//                         ctx.convert_float_to_uint(Some(c), x)
//                     }
//                     _ => unreachable!(),
//                 },
//                 _ => unreachable!(),
//             })
//         }
//         &SimplifiedExpression::Swizzle1(x, a) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             let spv::Type::Vector(component_type, _) = xt else {
//                 unreachable!();
//             };

//             Some((
//                 ctx.composite_extract(component_type.clone(), x, vec![a as _]),
//                 component_type.into(),
//             ))
//         }
//         &SimplifiedExpression::Swizzle2(x, a, b) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             let spv::Type::Vector(component_type, _) = xt else {
//                 unreachable!();
//             };

//             let rt = component_type.of_vector(spv::VectorSize::Two);
//             Some((
//                 ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _]),
//                 rt.into(),
//             ))
//         }
//         &SimplifiedExpression::Swizzle3(x, a, b, c) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             let spv::Type::Vector(component_type, _) = xt else {
//                 unreachable!();
//             };

//             let rt = component_type.of_vector(spv::VectorSize::Three);
//             Some((
//                 ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _]),
//                 rt.into(),
//             ))
//         }
//         &SimplifiedExpression::Swizzle4(x, a, b, c, d) => {
//             let (x, xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             let spv::Type::Vector(component_type, _) = xt else {
//                 unreachable!();
//             };

//             let rt = component_type.of_vector(spv::VectorSize::Four);
//             Some((
//                 ctx.vector_shuffle_1(rt.clone(), x, vec![a as _, b as _, c as _, d as _]),
//                 rt.into(),
//             ))
//         }
//         &SimplifiedExpression::VectorShuffle4(v1, v2, a, b, c, d) => {
//             let (v1, v1t) = emit_expr_spv_ops(body, v1.0, ctx).unwrap();
//             let (v2, _) = emit_expr_spv_ops(body, v2.0, ctx).unwrap();
//             let spv::Type::Vector(component_type, _) = v1t else {
//                 unreachable!("v1t = {v1t:?}");
//             };

//             let rt = component_type.of_vector(spv::VectorSize::Four);
//             Some((
//                 ctx.vector_shuffle(rt.clone(), v1, v2, vec![a as _, b as _, c as _, d as _]),
//                 rt.into(),
//             ))
//         }
//         SimplifiedExpression::CanonicalPathRef(rp) => {
//             match ctx.entry_point_maps.refpath_to_global_var.get(rp) {
//                 Some(&GlobalAccessType::Direct(gv, ref vty, storage_class)) => {
//                     Some((gv, vty.clone().of_pointer(storage_class)))
//                 }
//                 Some(&GlobalAccessType::PushConstantStruct {
//                     struct_var,
//                     member_index,
//                     ref member_ty,
//                 }) => {
//                     let member_ty = member_ty
//                         .clone()
//                         .of_pointer(spv::asm::StorageClass::PushConstant);

//                     let ac_index_id = ctx.module_ctx.request_const_id(member_index.into());
//                     Some((
//                         ctx.access_chain(member_ty.clone(), struct_var, vec![ac_index_id]),
//                         member_ty,
//                     ))
//                 }
//                 None => panic!("no corresponding canonical refpath found? {rp:?}"),
//             }
//         }
//         SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
//             match ctx.entry_point_maps.refpath_to_global_var.get(rp) {
//                 Some(GlobalAccessType::Direct(gv, vty, _)) => {
//                     let (gv, vty) = (gv.clone(), vty.clone());

//                     Some((ctx.load(vty.clone(), gv), vty))
//                 }
//                 Some(&GlobalAccessType::PushConstantStruct {
//                     struct_var,
//                     member_index,
//                     ref member_ty,
//                 }) => {
//                     let member_ty = member_ty.clone();

//                     let ac_index_id = ctx.module_ctx.request_const_id(member_index.into());
//                     Some((
//                         ctx.chained_load(
//                             member_ty.clone(),
//                             spv::asm::StorageClass::PushConstant,
//                             struct_var,
//                             vec![ac_index_id],
//                         ),
//                         member_ty,
//                     ))
//                 }
//                 // TODO: あとで再帰組む
//                 None => match rp {
//                     RefPath::Member(root, index) => {
//                         Some(match ctx.entry_point_maps.refpath_to_global_var.get(root) {
//                             Some(GlobalAccessType::Direct(gv, vty, storage_class)) => {
//                                 let (gv, vty) = (gv.clone(), vty.clone());
//                                 let member_ty = match vty {
//                                     spv::Type::Struct { member_types, .. } => {
//                                         member_types[*index].ty.clone()
//                                     }
//                                     _ => unreachable!("cannot member ref"),
//                                 };

//                                 let ac_index_id =
//                                     ctx.module_ctx.request_const_id((*index as u32).into());
//                                 (
//                                     ctx.chained_load(
//                                         member_ty.clone(),
//                                         *storage_class,
//                                         gv,
//                                         vec![ac_index_id],
//                                     ),
//                                     member_ty,
//                                 )
//                             }
//                             Some(&GlobalAccessType::PushConstantStruct {
//                                 struct_var,
//                                 member_index,
//                                 ref member_ty,
//                             }) => {
//                                 let member_ty = member_ty.clone();
//                                 let final_ty = match member_ty {
//                                     spv::Type::Struct { member_types, .. } => {
//                                         member_types[*index].ty.clone()
//                                     }
//                                     _ => unreachable!("cannot member ref"),
//                                 };

//                                 let ac_index_id =
//                                     ctx.module_ctx.request_const_id(member_index.into());
//                                 let ac_index2_id =
//                                     ctx.module_ctx.request_const_id((*index as u32).into());
//                                 (
//                                     ctx.chained_load(
//                                         final_ty.clone(),
//                                         spv::asm::StorageClass::PushConstant,
//                                         struct_var,
//                                         vec![ac_index_id, ac_index2_id],
//                                     ),
//                                     final_ty,
//                                 )
//                             }
//                             None => panic!("no corresponding canonical refpath found? {rp:?}"),
//                         })
//                     }
//                     _ => panic!("no corresponding canonical refpath found? {rp:?}"),
//                 },
//             }
//         }
//         SimplifiedExpression::ConstSInt(s, mods) => {
//             let mut x: i32 = match s
//                 .0
//                 .slice
//                 .strip_prefix("0x")
//                 .or_else(|| s.0.slice.strip_prefix("0X"))
//             {
//                 Some(left) => i32::from_str_radix(left, 16).unwrap(),
//                 None => s.0.slice.parse().unwrap(),
//             };
//             if mods.contains(ConstModifiers::NEGATE) {
//                 x = -x;
//             }
//             if mods.contains(ConstModifiers::BIT_NOT) {
//                 x = !x;
//             }
//             if mods.contains(ConstModifiers::LOGICAL_NOT) {
//                 x = if x == 0 { 1 } else { 0 };
//             }

//             Some((
//                 ctx.module_ctx.request_const_id(x.into()),
//                 spv::Type::sint(32),
//             ))
//         }
//         SimplifiedExpression::ConstUInt(s, mods) => {
//             let mut x: u32 = match s
//                 .0
//                 .slice
//                 .strip_prefix("0x")
//                 .or_else(|| s.0.slice.strip_prefix("0X"))
//             {
//                 Some(left) => u32::from_str_radix(left, 16).unwrap(),
//                 None => s.0.slice.parse().unwrap(),
//             };
//             if mods.contains(ConstModifiers::NEGATE) {
//                 panic!("applying negate to unsigned number?");
//             }
//             if mods.contains(ConstModifiers::BIT_NOT) {
//                 x = !x;
//             }
//             if mods.contains(ConstModifiers::LOGICAL_NOT) {
//                 x = if x == 0 { 1 } else { 0 };
//             }

//             Some((
//                 ctx.module_ctx.request_const_id(x.into()),
//                 spv::Type::uint(32),
//             ))
//         }
//         SimplifiedExpression::ConstFloat(s, mods) => {
//             let mut x: f32 =
//                 s.0.slice
//                     .strip_suffix(['f', 'F'])
//                     .unwrap_or(s.0.slice)
//                     .parse()
//                     .unwrap();
//             if mods.contains(ConstModifiers::NEGATE) {
//                 x = -x;
//             }
//             if mods.contains(ConstModifiers::BIT_NOT) {
//                 x = unsafe { core::mem::transmute(!core::mem::transmute::<_, u32>(x)) };
//             }
//             if mods.contains(ConstModifiers::LOGICAL_NOT) {
//                 x = if x != 0.0 { 0.0 } else { 1.0 };
//             }

//             Some((
//                 ctx.module_ctx.request_const_id(x.into()),
//                 spv::Type::float(32),
//             ))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.SubpassLoad", _, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();
//             assert!(arg_ty.len() == 1 && arg_ty[0] == spv::Type::SUBPASS_DATA_IMAGE_TYPE);

//             let rt = spv::Type::float(32).of_vector(4);
//             let result_type = ctx.module_ctx.request_type_id(rt.clone());
//             let result_id = ctx.new_id();
//             let coordinate_const = ctx
//                 .module_ctx
//                 .request_const_id(spv::Constant::i32vec2(0, 0));
//             ctx.ops.push(spv::Instruction::ImageRead {
//                 result_type,
//                 result: result_id,
//                 image: args[0],
//                 coordinate: coordinate_const,
//             });
//             Some((result_id, rt))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Dot#Float3", _, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();
//             assert!(
//                 arg_ty.len() == 2
//                     && arg_ty
//                         .iter()
//                         .all(|t| *t == spv::Type::float(32).of_vector(3))
//             );

//             let rt = spv::Type::float(32);
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::Dot {
//                 result_type,
//                 result,
//                 vector1: args[0],
//                 vector2: args[1],
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Normalize#Float3", _, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();
//             assert_eq!(arg_ty.len(), 1);
//             assert_eq!(arg_ty[0], spv::Type::float(32).of_vector(3));

//             let glsl_std_450_ext_set = ctx.module_ctx.request_ext_inst_set("GLSL.std.450".into());

//             let rt = spv::Type::float(32).of_vector(3);
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::ExtInst {
//                 result_type,
//                 result,
//                 set: glsl_std_450_ext_set,
//                 instruction: 69,
//                 operands: args,
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.Transpose#Float4x4", _, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();
//             assert_eq!(arg_ty.len(), 1);
//             assert_eq!(arg_ty[0], spv::Type::float(32).of_matrix(4, 4));

//             let rt = spv::Type::float(32).of_matrix(4, 4);
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::Transpose {
//                 result_type,
//                 result,
//                 matrix: args[0],
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.SampleAt#Texture2D", _, args) => {
//             let (args, arg_ty): (Vec<_>, Vec<_>) = args
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap())
//                 .unzip();
//             assert_eq!(arg_ty.len(), 2);
//             assert!(matches!(arg_ty[0], spv::Type::SampledImage { .. }));
//             assert_eq!(arg_ty[1], spv::Type::float(32).of_vector(2));

//             let rt = spv::Type::float(32).of_vector(4);
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::ImageSampleImplicitLod {
//                 result_type,
//                 result,
//                 sampled_image: args[0],
//                 coordinate: args[1],
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.ExecutionBarrier", _, args) => {
//             assert!(args.is_empty());

//             // とりあえずGLSLに倣って固定値で出してるけどこれでいいんだろうか
//             let execution_scope = ctx
//                 .module_ctx
//                 .request_const_id(spv::Constant::from(spv::asm::Scope::Workgroup as u32));
//             let memory_scope = ctx
//                 .module_ctx
//                 .request_const_id(spv::Constant::from(spv::asm::Scope::Workgroup as u32));
//             let semantics = ctx.module_ctx.request_const_id(spv::Constant::from(
//                 (spv::asm::MemorySemantics::ACQUIRE_RELEASE
//                     | spv::asm::MemorySemantics::WORKGROUP_MEMORY)
//                     .bits(),
//             ));
//             ctx.ops.push(spv::Instruction::ControlBarrier {
//                 execution: execution_scope,
//                 memory: memory_scope,
//                 semantics,
//             });
//             None
//         }
//         SimplifiedExpression::ChainedRef(base, xs) => {
//             let (base, bty) = emit_expr_spv_ops(body, base.0, ctx).unwrap();
//             let xs = xs
//                 .iter()
//                 .map(|x| emit_expr_spv_ops(body, x.0, ctx).unwrap().0)
//                 .collect::<Vec<_>>();

//             let spv::Type::Pointer(p) = bty else {
//                 unreachable!()
//             };
//             let term_type = match body[expr_id].1 {
//                 ConcreteType::Ref(ref inner) => &**inner,
//                 ConcreteType::MutableRef(ref inner) => &**inner,
//                 _ => unreachable!(),
//             };

//             // TODO: ここの型生成方法あとで考える
//             let rt = term_type
//                 .clone()
//                 .make_spv_type(&SymbolScope::new_intrinsics())
//                 .of_pointer(p.storage_class);
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::AccessChain {
//                 base,
//                 result_type,
//                 result,
//                 indexes: xs,
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::LoadRef(r) => {
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();

//             let rt = rt.dereferenced().unwrap();
//             let (result_type, result) = ctx.issue_typed_ids(rt.clone());
//             ctx.ops.push(spv::Instruction::Load {
//                 result_type,
//                 result,
//                 pointer: r,
//             });
//             Some((result, rt))
//         }
//         SimplifiedExpression::StoreRef(r, v) => {
//             let (r, rt) = emit_expr_spv_ops(body, r.0, ctx).unwrap();
//             let (v, vt) = emit_expr_spv_ops(body, v.0, ctx).unwrap();
//             assert_eq!(rt.dereferenced(), Some(vt));
//             ctx.ops.push(spv::Instruction::Store {
//                 pointer: r,
//                 object: v,
//             });

//             None
//         }
//         SimplifiedExpression::StoreOutput(x, o) => {
//             let (x, _xt) = emit_expr_spv_ops(body, x.0, ctx).unwrap();
//             ctx.ops.push(spv::Instruction::Store {
//                 pointer: ctx.entry_point_maps.output_global_vars[*o],
//                 object: x,
//             });

//             None
//         }
//         SimplifiedExpression::ScopedBlock {
//             expressions,
//             returning,
//             ..
//         } => emit_expr_spv_ops(expressions, returning.0, ctx),
//         x => unimplemented!("{x:?}"),
//     };

//     ctx.emitted_expression_id.insert(expr_id, result.clone());
//     result
// }
