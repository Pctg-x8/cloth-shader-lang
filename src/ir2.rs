use crate::concrete_type::IntrinsicType;
use crate::ir::block::BlockFlowInstruction;
use crate::ir::{ConstIntLiteral, ConstNumberLiteral};
use crate::spirv as spv;
use std::collections::HashMap;
use std::fmt::Write;

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

#[repr(transparent)]
pub struct IndentWriter(pub usize);
impl core::fmt::Display for IndentWriter {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for x in 0..self.0 {
            f.write_str("  ")?;
        }

        Ok(())
    }
}

#[repr(transparent)]
pub struct SwizzleElementWriter<'s>(pub &'s [usize]);
impl core::fmt::Display for SwizzleElementWriter<'_> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for x in self.0 {
            f.write_char(['x', 'y', 'z', 'w'][*x])?;
        }

        Ok(())
    }
}

#[repr(transparent)]
pub struct CommaSeparatedWriter<'s, T: 's>(pub &'s [T]);
impl<'s, T: 's> core::fmt::Debug for CommaSeparatedWriter<'s, T>
where
    T: core::fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        for x in self.0 {
            if wrote {
                f.write_str(", ")?;
            }
            <T as core::fmt::Debug>::fmt(x, f)?;
            wrote = true;
        }

        Ok(())
    }
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
    BlockValue(ConstOrInst<'s>),
    ContinueLoop,
    BreakLoop,
    Return(ConstOrInst<'s>),
}
impl<'s> Inst<'s> {
    pub fn dump_list(
        list: &[Self],
        sink: &mut (impl std::io::Write + ?Sized),
    ) -> std::io::Result<()> {
        fn dump(
            x: &Inst,
            sink: &mut (impl std::io::Write + ?Sized),
            depth: usize,
        ) -> std::io::Result<()> {
            let indent = IndentWriter(depth);

            match x {
                Inst::BuiltinIORef(b) => writeln!(sink, "{indent}BuiltinIORef {b:?}"),
                Inst::DescriptorRef { set, binding } => {
                    writeln!(sink, "{indent}DescriptorRef set={set} binding={binding}")
                }
                Inst::PushConstantRef(o) => writeln!(sink, "{indent}PushConstantRef offset={o}"),
                Inst::WorkgroupSharedMemoryRef(rp) => {
                    writeln!(sink, "{indent}WorkgroupSharedMemoryRef id={rp:?}")
                }
                Inst::CompositeRef(src, compo) => {
                    writeln!(sink, "{indent}CompositeRef {src:?} {compo:?}")
                }
                Inst::IntrinsicBinary(left, op, right) => {
                    writeln!(sink, "{indent}{left:?} `{op:?}` {right:?}")
                }
                Inst::IntrinsicUnary(value, op) => writeln!(sink, "{indent}unary-{op:?} {value:?}"),
                Inst::Cast(src, t) => writeln!(sink, "{indent}{src:?} as {t:?}"),
                Inst::ConstructIntrinsicType(it, xs) => {
                    writeln!(sink, "{indent}{it:?}({:?})", CommaSeparatedWriter(xs))
                }
                Inst::ConstructComposite(xs) => {
                    writeln!(sink, "{indent}composite({:?})", CommaSeparatedWriter(xs))
                }
                Inst::Swizzle(src, xs) => {
                    writeln!(sink, "{indent}{src:?}.{}", SwizzleElementWriter(xs))
                }
                Inst::LoadRef(ptr) => writeln!(sink, "{indent}Load {ptr:?}"),
                Inst::StoreRef { ptr, value } => {
                    writeln!(sink, "{indent}Store {ptr:?} <- {value:?}")
                }
                Inst::IntrinsicCall { identifier, args } => writeln!(
                    sink,
                    "{indent}IntrinsicCall {identifier}({:?})",
                    CommaSeparatedWriter(args)
                ),
                Inst::Branch {
                    source,
                    true_instructions,
                    false_instructions,
                } => {
                    writeln!(sink, "{indent}Branch if {source:?} {{")?;
                    for x in true_instructions.iter() {
                        dump(x, sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}} else {{")?;
                    for x in false_instructions.iter() {
                        dump(x, sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}}")
                }
                Inst::LoopWhile { condition, body } => {
                    writeln!(sink, "{indent}Loop while {condition:?} {{")?;
                    for x in body.iter() {
                        dump(x, sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}}")
                }
                Inst::ContinueLoop => writeln!(sink, "{indent}continue"),
                Inst::BreakLoop => writeln!(sink, "{indent}break"),
                Inst::Return(v) => writeln!(sink, "{indent}return {v:?}"),
                Inst::BlockValue(v) => writeln!(sink, "{indent}block value = {v:?}"),
            }
        }

        for x in list {
            dump(x, sink, 0)?;
        }

        Ok(())
    }
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
            &BlockInstruction::ConstInt(ConstIntLiteral(ref s, m)) => {
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Const(ConstValue::SIntLit(ConstSIntLiteral(s.clone(), m))),
                );
            }
            &BlockInstruction::ConstNumber(ConstNumberLiteral(ref s, m)) => {
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Const(ConstValue::FloatLit(ConstFloatLiteral(s.clone(), m))),
                );
            }
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
            &BlockInstruction::ImmInt(v) => {
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Const(ConstValue::SInt(v.try_into().unwrap())),
                );
            }
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
            BlockInstruction::IntrinsicFunctionRef(_) => {
                unreachable!("non stripped intrinsic function ref")
            }
            BlockInstruction::UserDefinedFunctionRef(_, _) => {
                unreachable!("non inlined user defined function")
            }
            BlockInstruction::IntrinsicTypeConstructorRef(_) => {
                unreachable!("non stripped intrinsic type ctor ref")
            }
            &BlockInstruction::BuiltinIORef(b) => {
                function.instructions.push(Inst::BuiltinIORef(b));
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                );
            }
            &BlockInstruction::PushConstantRef(o) => {
                function.instructions.push(Inst::PushConstantRef(o));
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                );
            }
            &BlockInstruction::DescriptorRef { set, binding } => {
                function
                    .instructions
                    .push(Inst::DescriptorRef { set, binding });
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                );
            }
            &BlockInstruction::WorkgroupSharedMemoryRef(ref p) => {
                function
                    .instructions
                    .push(Inst::WorkgroupSharedMemoryRef(p.clone()));
                register_to_inst_const_map.insert(
                    r,
                    ConstOrInst::Inst(InstRef(function.instructions.len() - 1)),
                );
            }
            BlockInstruction::Cast(_, _)
            | BlockInstruction::InstantiateIntrinsicTypeClass(_, _)
            | BlockInstruction::LoadRef(_)
            | BlockInstruction::ConstructIntrinsicComposite(_, _)
            | BlockInstruction::ConstructTuple(_)
            | BlockInstruction::ConstructStruct(_)
            | BlockInstruction::PromoteIntToNumber(_)
            | BlockInstruction::IntrinsicBinaryOp(_, _, _)
            | BlockInstruction::IntrinsicUnaryOp(_, _)
            | BlockInstruction::Swizzle(_, _)
            | BlockInstruction::SwizzleRef(_, _)
            | BlockInstruction::StaticPathRef(_)
            | BlockInstruction::MemberRef(_, _)
            | BlockInstruction::TupleRef(_, _)
            | BlockInstruction::ArrayRef { .. }
            | BlockInstruction::Phi(_)
            | BlockInstruction::RegisterAlias(_)
            | BlockInstruction::PureIntrinsicCall(_, _)
            | BlockInstruction::PureFuncall(_, _) => {
                unreachable!("Not a const instruction {x:?}")
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
        until: Option<crate::ir::block::BlockRef>,
        last_block: &mut crate::ir::block::BlockRef,
        register_to_inst_const_map: &mut HashMap<RegisterRef, ConstOrInst<'s>>,
        instructions: &mut Vec<Inst<'s>>,
    ) {
        if until.is_some_and(|x| x == n) {
            // ここまで
            return;
        }

        let mut generated = true;
        while generated {
            generated = false;

            for (&r, x) in blocks[n.0].instructions.iter() {
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
                            instructions.push(Inst::Cast(i, ty.clone()));
                            register_to_inst_const_map
                                .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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
                            instructions.push(Inst::LoadRef(i));
                            register_to_inst_const_map
                                .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::ConstructIntrinsicType(it, arg_refs));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::ConstructComposite(arg_refs));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::ConstructComposite(arg_refs));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
                        generated = true;
                    }
                    &BlockInstruction::IntrinsicBinaryOp(left, op, right) => 'try_emit: {
                        let Some(left) = register_to_inst_const_map.get(&left) else {
                            break 'try_emit;
                        };
                        let Some(right) = register_to_inst_const_map.get(&right) else {
                            break 'try_emit;
                        };

                        instructions.push(Inst::IntrinsicBinary(left.clone(), op, right.clone()));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
                        generated = true;
                    }
                    &BlockInstruction::IntrinsicUnaryOp(value, op) => 'try_emit: {
                        let Some(value) = register_to_inst_const_map.get(&value) else {
                            break 'try_emit;
                        };

                        instructions.push(Inst::IntrinsicUnary(value.clone(), op));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::Swizzle(src.clone(), elements.clone()));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                            instructions.push(Inst::CompositeRef(
                                i,
                                ConstOrInst::Const(ConstValue::UInt(member_index as _)),
                            ));
                            register_to_inst_const_map
                                .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
                            generated = true;
                        }
                        Some(&ConstOrInst::Const(_)) => unimplemented!("member ref of const value"),
                        None => (),
                    },
                    &BlockInstruction::TupleRef(src, index) => {
                        match register_to_inst_const_map.get(&src) {
                            Some(&ConstOrInst::Inst(i)) => {
                                instructions.push(Inst::CompositeRef(
                                    i,
                                    ConstOrInst::Const(ConstValue::UInt(index as _)),
                                ));
                                register_to_inst_const_map
                                    .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::CompositeRef(source, index.clone()));
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
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

                        instructions.push(Inst::IntrinsicCall {
                            identifier,
                            args: arg_refs,
                        });
                        register_to_inst_const_map
                            .insert(r, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
                        generated = true;
                    }
                    BlockInstruction::PureFuncall(_, _) => {
                        unreachable!("all user defined functions inlined")
                    }
                }
            }
        }

        *last_block = n;

        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                process(
                    blocks,
                    registers,
                    const_map,
                    next,
                    until,
                    last_block,
                    register_to_inst_const_map,
                    instructions,
                );
            }
            BlockFlowInstruction::StoreRef { ptr, value, after } => {
                let Some(&ConstOrInst::Inst(ptr)) = register_to_inst_const_map.get(&ptr) else {
                    eprintln!("ptr not found: {:#?}", register_to_inst_const_map);
                    unreachable!("b{} r{} {:?}", n.0, ptr.0, blocks[n.0].flow);
                };
                let Some(value) = register_to_inst_const_map.get(&value).cloned() else {
                    unreachable!();
                };
                instructions.push(Inst::StoreRef { ptr, value });

                process(
                    blocks,
                    registers,
                    const_map,
                    after.unwrap(),
                    until,
                    last_block,
                    register_to_inst_const_map,
                    instructions,
                );
            }
            BlockFlowInstruction::Funcall {
                callee,
                ref args,
                result,
                after_return,
            } => {
                let Some(callee) = register_to_inst_const_map.get(&callee) else {
                    eprintln!("callee not found: {:#?}", register_to_inst_const_map);
                    unreachable!("b{} r{} {:?}", n.0, callee.0, blocks[n.0].flow);
                };
                let mut arg_refs = Vec::with_capacity(args.len());
                for a in args.iter() {
                    arg_refs.push(register_to_inst_const_map.get(a).unwrap());
                }
                unimplemented!("FunctionCall {callee:?}");
            }
            BlockFlowInstruction::IntrinsicImpureFuncall {
                identifier,
                ref args,
                result,
                after_return,
            } => {
                let mut arg_refs = Vec::with_capacity(args.len());
                for a in args.iter() {
                    arg_refs.push(register_to_inst_const_map.get(a).unwrap().clone());
                }

                instructions.push(Inst::IntrinsicCall {
                    identifier,
                    args: arg_refs,
                });
                register_to_inst_const_map
                    .insert(result, ConstOrInst::Inst(InstRef(instructions.len() - 1)));
                process(
                    blocks,
                    registers,
                    const_map,
                    after_return.unwrap(),
                    until,
                    last_block,
                    register_to_inst_const_map,
                    instructions,
                );
            }
            BlockFlowInstruction::Conditional {
                source,
                r#true,
                r#false,
                merge,
            } => {
                let Some(source) = register_to_inst_const_map.get(&source).cloned() else {
                    eprintln!("source not found: {:#?}", register_to_inst_const_map);
                    unreachable!("b{} r{} {:?}", n.0, source.0, blocks[n.0].flow);
                };
                let mut true_instructions = Vec::new();
                let mut true_last_block = n;
                process(
                    blocks,
                    registers,
                    const_map,
                    r#true,
                    Some(merge),
                    &mut true_last_block,
                    register_to_inst_const_map,
                    &mut true_instructions,
                );
                let mut false_instructions = Vec::new();
                let mut false_last_block = n;
                process(
                    blocks,
                    registers,
                    const_map,
                    r#false,
                    Some(merge),
                    &mut false_last_block,
                    register_to_inst_const_map,
                    &mut false_instructions,
                );

                if let Some((merge_phi_register, merge_phi_incomings)) = blocks[merge.0]
                    .instructions
                    .iter()
                    .find_map(|(r, x)| match x {
                        BlockInstruction::Phi(incomings) => Some((*r, incomings)),
                        _ => None,
                    })
                {
                    let Some(true_phi_incoming_register) =
                        merge_phi_incomings.get(&true_last_block)
                    else {
                        unreachable!(
                            "no true phi incoming register true_last_blk=b{}",
                            true_last_block.0
                        );
                    };
                    let Some(false_phi_incoming_register) =
                        merge_phi_incomings.get(&false_last_block)
                    else {
                        unreachable!(
                            "no false phi incoming register false_last_blk=b{}",
                            false_last_block.0
                        );
                    };
                    let Some(true_phi_value) =
                        register_to_inst_const_map.get(true_phi_incoming_register)
                    else {
                        unreachable!("no value for true phi");
                    };
                    let Some(false_phi_value) =
                        register_to_inst_const_map.get(false_phi_incoming_register)
                    else {
                        unreachable!("no value for false phi");
                    };

                    true_instructions.push(Inst::BlockValue(true_phi_value.clone()));
                    false_instructions.push(Inst::BlockValue(false_phi_value.clone()));
                    instructions.push(Inst::Branch {
                        source,
                        true_instructions,
                        false_instructions,
                    });
                    register_to_inst_const_map.insert(
                        merge_phi_register,
                        ConstOrInst::Inst(InstRef(instructions.len() - 1)),
                    );
                } else {
                    instructions.push(Inst::Branch {
                        source,
                        true_instructions,
                        false_instructions,
                    });
                }

                process(
                    blocks,
                    registers,
                    const_map,
                    merge,
                    until,
                    last_block,
                    register_to_inst_const_map,
                    instructions,
                );
            }
            BlockFlowInstruction::ConditionalLoop {
                condition,
                r#break,
                r#continue,
                body,
            } => {
                let Some(condition) = register_to_inst_const_map.get(&condition).cloned() else {
                    unreachable!();
                };

                let mut body_instructions = Vec::new();
                let mut body_last_block = body;
                process(
                    blocks,
                    registers,
                    const_map,
                    body,
                    Some(r#break),
                    &mut body_last_block,
                    register_to_inst_const_map,
                    &mut body_instructions,
                );

                instructions.push(Inst::LoopWhile {
                    condition,
                    body: body_instructions,
                });
                process(
                    blocks,
                    registers,
                    const_map,
                    r#break,
                    until,
                    last_block,
                    register_to_inst_const_map,
                    instructions,
                );
            }
            BlockFlowInstruction::Break => {
                instructions.push(Inst::BreakLoop);
            }
            BlockFlowInstruction::Continue => {
                instructions.push(Inst::ContinueLoop);
            }
            BlockFlowInstruction::Return(v) => {
                let Some(v) = register_to_inst_const_map.get(&v).cloned() else {
                    unreachable!();
                };

                instructions.push(Inst::Return(v));
            }
            BlockFlowInstruction::Undetermined => unreachable!("undetermined destination"),
        }
    }

    process(
        blocks,
        registers,
        const_map,
        crate::ir::block::BlockRef(0),
        None,
        &mut crate::ir::block::BlockRef(0),
        &mut register_to_inst_const_map,
        &mut function.instructions,
    );

    function
}
