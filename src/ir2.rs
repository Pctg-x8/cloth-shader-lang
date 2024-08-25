use crate::concrete_type::IntrinsicType;
use crate::spirv as spv;
use std::collections::HashMap;

use crate::symbol::meta::BuiltinInputOutput;
use crate::{
    concrete_type::ConcreteType,
    ir::{
        block::{BlockInstruction, IntrinsicBinaryOperation, IntrinsicUnaryOperation, RegisterRef},
        ConstFloatLiteral, ConstSIntLiteral, ConstUIntLiteral,
    },
    ref_path::RefPath,
};

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct InstRef(pub usize);
impl std::fmt::Debug for InstRef {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub enum ConstValue<'s> {
    Unit,
    True,
    False,
    UInt(u32),
    SInt(i32),
    Float(f32),
    UIntLit(ConstUIntLiteral<'s>),
    SIntLit(ConstSIntLiteral<'s>),
    FloatLit(ConstFloatLiteral<'s>),
}

#[derive(Debug, Clone)]
pub enum ConstOrInst<'s> {
    Const(ConstValue<'s>),
    Inst(InstRef),
}

#[derive(Debug, Clone)]
pub enum Inst<'s> {
    BuiltinIORef(BuiltinInputOutput),
    DescriptorRef {
        set: u32,
        binding: u32,
    },
    PushConstantRef(u32),
    WorkgroupSharedMemoryRef(RefPath),
    CompositeRef(InstRef, ConstOrInst<'s>),
    IntrinsicBinary(ConstOrInst<'s>, IntrinsicBinaryOperation, ConstOrInst<'s>),
    IntrinsicUnary(ConstOrInst<'s>, IntrinsicUnaryOperation),
    Cast(InstRef, ConcreteType<'s>),
    ConstructIntrinsicType(IntrinsicType, Vec<ConstOrInst<'s>>),
    ConstructComposite(Vec<ConstOrInst<'s>>),
    Swizzle(ConstOrInst<'s>, Vec<usize>),
    LoadRef(InstRef),
    StoreRef {
        ptr: InstRef,
        value: ConstOrInst<'s>,
    },
    IntrinsicCall {
        identifier: &'static str,
        args: Vec<ConstOrInst<'s>>,
    },
    Branch {
        source: ConstOrInst<'s>,
        true_instructions: Vec<Inst<'s>>,
        false_instructions: Vec<Inst<'s>>,
    },
    LoopWhile {
        condition: ConstOrInst<'s>,
        body: Vec<Inst<'s>>,
    },
    ContinueLoop,
    BreakLoop,
    Return(ConstOrInst<'s>),
}

pub struct Function<'s> {
    pub instructions: Vec<Inst<'s>>,
}

pub fn reconstruct<'a, 's>(
    blocks: &[crate::ir::block::Block<'a, 's>],
    registers: &[ConcreteType<'s>],
    const_map: &HashMap<crate::ir::block::RegisterRef, crate::ir::block::BlockInstruction<'a, 's>>,
) -> Function<'s> {
    let mut function = Function {
        instructions: Vec::new(),
    };
    let mut register_to_inst_const_map = HashMap::new();

    for (&r, x) in const_map.iter() {
        match x {
            BlockInstruction::ConstUnit => {
                register_to_inst_const_map.insert(r, ConstOrInst::Const(ConstValue::Unit));
            }
            BlockInstruction::ConstInt(_) => unreachable!("uninstantiated const int"),
            BlockInstruction::ConstNumber(_) => unreachable!("uninstantiated const number"),
            BlockInstruction::ConstSInt(l) => {
                register_to_inst_const_map
                    .insert(r, ConstOrInst::Const(ConstValue::SIntLit(l.clone())));
            }
            BlockInstruction::ConstUInt(l) => {
                register_to_inst_const_map
                    .insert(r, ConstOrInst::Const(ConstValue::UIntLit(l.clone())));
            }
            BlockInstruction::ConstFloat(l) => {
                register_to_inst_const_map
                    .insert(r, ConstOrInst::Const(ConstValue::FloatLit(l.clone())));
            }
            BlockInstruction::ImmBool(true) => {
                register_to_inst_const_map.insert(r, ConstOrInst::Const(ConstValue::True));
            }
            BlockInstruction::ImmBool(false) => {
                register_to_inst_const_map.insert(r, ConstOrInst::Const(ConstValue::False));
            }
            BlockInstruction::ImmInt(_) => unreachable!("uninstantiated imm int"),
            &BlockInstruction::ImmSInt(v) => {
                register_to_inst_const_map.insert(r, ConstOrInst::Const(ConstValue::SInt(v)));
            }
            &BlockInstruction::ImmUInt(v) => {
                register_to_inst_const_map.insert(r, ConstOrInst::Const(ConstValue::UInt(v)));
            }
            BlockInstruction::ScopeLocalVarRef(_, _) => {
                unreachable!("non strippped scope local var")
            }
            BlockInstruction::FunctionInputVarRef(_, _) => {
                unreachable!("non strippped function input var")
            }
        }
    }

    fn process<'a, 's>(
        blocks: &[crate::ir::block::Block<'a, 's>],
        registers: &[ConcreteType<'s>],
        const_map: &HashMap<
            crate::ir::block::RegisterRef,
            crate::ir::block::BlockInstruction<'a, 's>,
        >,
        n: crate::ir::block::BlockRef,
        function: &mut Function<'s>,
    ) {
        let mut register_to_inst_const_map = HashMap::new();

        let mut generated = true;
        while generated {
            generated = false;

            for (r, x) in blocks[n.0].instructions.iter() {
                if register_to_inst_const_map.contains_key(&r) {
                    // 生成済み
                    continue;
                }

                match x {
                    BlockInstruction::ConstUnit => {
                        unreachable!("ConstUnit cannot be appeared in block")
                    }
                    BlockInstruction::ConstFloat(_) => {
                        unreachable!("ConstFloat cannot be appeared in block")
                    }
                    BlockInstruction::ConstSInt(_) => {
                        unreachable!("ConstSInt cannot be appeared in block")
                    }
                    BlockInstruction::ConstUInt(_) => {
                        unreachable!("ConstUInt cannot be appeared in block")
                    }
                    BlockInstruction::ImmBool(_) => {
                        unreachable!("ImmBool cannot be appeared in block")
                    }
                    &BlockInstruction::ImmSInt(_) => {
                        unreachable!("ImmSInt cannot be appeared in block")
                    }
                    &BlockInstruction::ImmUInt(_) => {
                        unreachable!("ImmUInt cannot be appeared in block")
                    }
                    BlockInstruction::ImmInt(_) => unreachable!("non instantiated imm int"),
                    BlockInstruction::ConstInt(_) => unreachable!("non instantiated int lit"),
                    BlockInstruction::ConstNumber(_) => unreachable!("non instantiated number lit"),
                    BlockInstruction::PromoteIntToNumber(_) => {
                        unreachable!("unprocessed promotion int -> number")
                    }
                    BlockInstruction::BuiltinIORef(_) => {
                        unreachable!("BuiltinIORef cannot be appeared in block")
                    }
                    BlockInstruction::DescriptorRef { .. } => {
                        unreachable!("DescriptorRef cannot be appeared in block")
                    }
                    BlockInstruction::PushConstantRef(_) => {
                        unreachable!("PushConstantRef cannot be appeared in block")
                    }
                    BlockInstruction::WorkgroupSharedMemoryRef(_) => {
                        unreachable!("WorkgroupSharedMemoryRef cannot be appeared in block")
                    }
                    BlockInstruction::StaticPathRef(_) => unreachable!("deprecated"),
                    BlockInstruction::Cast(src, ty) => match register_to_inst_const_map.get(&src) {
                        Some(&ConstOrInst::Inst(i)) => {
                            function.instructions.push(Inst::Cast(i, ty.clone()));
                            register_to_inst_const_map.insert(
                                r,
                                ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                            );
                            generated = true;
                        }
                        Some(ConstOrInst::Const(_)) => unimplemented!("reduce const cast"),
                        _ => (),
                    },
                    BlockInstruction::InstantiateIntrinsicTypeClass(_, _) => {
                        unreachable!("non instantiated inst")
                    }
                    BlockInstruction::LoadRef(ptr) => match register_to_inst_const_map.get(&ptr) {
                        Some(&ConstOrInst::Inst(i)) => {
                            function.instructions.push(Inst::LoadRef(i));
                            register_to_inst_const_map.insert(
                                r,
                                ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                            );
                            generated = true;
                        }
                        Some(ConstOrInst::Const(_)) => unreachable!("cannot load ref of const"),
                        _ => (),
                    },
                    &BlockInstruction::ConstructIntrinsicComposite(it, ref args) => 'try_emit: {
                        let mut arg_refs = Vec::with_capacity(args.len());
                        for a in args.iter() {
                            let Some(x) = register_to_inst_const_map.get(a) else {
                                break 'try_emit;
                            };

                            arg_refs.push(x.clone());
                        }

                        function
                            .instructions
                            .push(Inst::ConstructIntrinsicType(it, arg_refs));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::ConstructTuple(xs) => 'try_emit: {
                        let mut arg_refs = Vec::with_capacity(xs.len());
                        for x in xs.iter() {
                            let Some(x) = register_to_inst_const_map.get(x) else {
                                break 'try_emit;
                            };

                            arg_refs.push(x.clone());
                        }

                        function
                            .instructions
                            .push(Inst::ConstructComposite(arg_refs));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::ConstructStruct(members) => 'try_emit: {
                        let mut arg_refs = Vec::with_capacity(members.len());
                        for x in members.iter() {
                            let Some(x) = register_to_inst_const_map.get(x) else {
                                break 'try_emit;
                            };

                            arg_refs.push(x.clone());
                        }

                        function
                            .instructions
                            .push(Inst::ConstructComposite(arg_refs));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    &BlockInstruction::IntrinsicBinaryOp(left, op, right) => 'try_emit: {
                        let Some(left) = register_to_inst_const_map.get(&left) else {
                            break 'try_emit;
                        };
                        let Some(right) = register_to_inst_const_map.get(&right) else {
                            break 'try_emit;
                        };

                        function.instructions.push(Inst::IntrinsicBinary(
                            left.clone(),
                            op,
                            right.clone(),
                        ));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    &BlockInstruction::IntrinsicUnaryOp(value, op) => 'try_emit: {
                        let Some(value) = register_to_inst_const_map.get(&value) else {
                            break 'try_emit;
                        };

                        function
                            .instructions
                            .push(Inst::IntrinsicUnary(value.clone(), op));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::IntrinsicFunctionRef(_) => {
                        unreachable!("unresolved intrinsic function ref")
                    }
                    BlockInstruction::IntrinsicTypeConstructorRef(_) => {
                        unreachable!("unresolved intrinsic tycon ref")
                    }
                    BlockInstruction::ScopeLocalVarRef(_, _) => {
                        unreachable!("scope local vars are stripped")
                    }
                    BlockInstruction::FunctionInputVarRef(_, _) => {
                        unreachable!("function input vars are stripped")
                    }
                    BlockInstruction::UserDefinedFunctionRef(_, _) => {
                        unreachable!("all user defined functions inlined")
                    }
                    BlockInstruction::Swizzle(src, elements) => 'try_emit: {
                        let Some(src) = register_to_inst_const_map.get(&src) else {
                            break 'try_emit;
                        };

                        function
                            .instructions
                            .push(Inst::Swizzle(src.clone(), elements.clone()));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::SwizzleRef(_, _) => unimplemented!("Swizzle Ref"),
                    &BlockInstruction::MemberRef(src, ref name) => match register_to_inst_const_map
                        .get(&src)
                    {
                        Some(&ConstOrInst::Inst(i)) => {
                            let member_index = match registers[src.0] {
                                ConcreteType::Ref(ref r) | ConcreteType::MutableRef(ref r) => {
                                    match &**r {
                                        ConcreteType::Struct(members) => {
                                            members.iter().position(|m| m.name == *name).unwrap()
                                        }
                                        _ => unreachable!("cannot ref member of this type"),
                                    }
                                }
                                _ => unreachable!("cannot make ref from this type"),
                            };

                            function.instructions.push(Inst::CompositeRef(
                                i,
                                ConstOrInst::Const(ConstValue::UInt(member_index as _)),
                            ));
                            register_to_inst_const_map.insert(
                                r,
                                ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                            );
                            generated = true;
                        }
                        Some(&ConstOrInst::Const(_)) => unimplemented!("member ref of const value"),
                        None => (),
                    },
                    &BlockInstruction::TupleRef(src, index) => {
                        match register_to_inst_const_map.get(&src) {
                            Some(&ConstOrInst::Inst(i)) => {
                                function.instructions.push(Inst::CompositeRef(
                                    i,
                                    ConstOrInst::Const(ConstValue::UInt(index as _)),
                                ));
                                register_to_inst_const_map.insert(
                                    r,
                                    ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                                );
                                generated = true;
                            }
                            Some(&ConstOrInst::Const(_)) => {
                                unimplemented!("member ref of const value")
                            }
                            None => (),
                        }
                    }
                    BlockInstruction::ArrayRef { source, index } => 'try_emit: {
                        let source = match register_to_inst_const_map.get(&source) {
                            Some(&ConstOrInst::Inst(i)) => i,
                            Some(&ConstOrInst::Const(_)) => {
                                unimplemented!("array ref of const value")
                            }
                            None => break 'try_emit,
                        };
                        let Some(index) = register_to_inst_const_map.get(&index) else {
                            break 'try_emit;
                        };

                        function
                            .instructions
                            .push(Inst::CompositeRef(source, index.clone()));
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::Phi(_) => unimplemented!("phi decoding"),
                    BlockInstruction::RegisterAlias(_) => unreachable!("Unresolved register alias"),
                    BlockInstruction::PureIntrinsicCall(identifier, args) => 'try_emit: {
                        let mut arg_refs = Vec::with_capacity(args.len());
                        for a in args.iter() {
                            let Some(x) = register_to_inst_const_map.get(a) else {
                                break 'try_emit;
                            };

                            arg_refs.push(x.clone());
                        }

                        function.instructions.push(Inst::IntrinsicCall {
                            identifier,
                            args: arg_refs,
                        });
                        register_to_inst_const_map.insert(
                            r,
                            ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                        );
                        generated = true;
                    }
                    BlockInstruction::PureFuncall(_, _) => {
                        unreachable!("all user defined functions inlined")
                    }
                }
            }
        }
    }

    process(
        blocks,
        registers,
        const_map,
        crate::ir::block::BlockRef(0),
        &mut function,
    );

    function
}
