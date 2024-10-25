use crate::concrete_type::IntrinsicType;
use crate::ir::block::{
    BlockConstInstruction, BlockFlowInstruction, BlockPureInstruction, BlockRef, BlockifiedProgram,
};
use crate::ir::{ConstIntLiteral, ConstNumberLiteral};
use crate::spirv as spv;
use std::collections::{BTreeMap, HashMap, HashSet};
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
impl From<InstRef> for ConstOrInst<'_> {
    #[inline(always)]
    fn from(value: InstRef) -> Self {
        Self::Inst(value)
    }
}

#[repr(transparent)]
pub struct IndentWriter(pub usize);
impl core::fmt::Display for IndentWriter {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.0 {
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
    CompositeInsert {
        source: ConstOrInst<'s>,
        index: usize,
        value: ConstOrInst<'s>,
    },
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
        true_instructions: HashMap<InstRef, Inst<'s>>,
        true_instruction_order: Vec<InstRef>,
        false_instructions: HashMap<InstRef, Inst<'s>>,
        false_instruction_order: Vec<InstRef>,
    },
    LoopWhile {
        incoming_block: BlockRef,
        condition: HashMap<InstRef, Inst<'s>>,
        condition_order: Vec<InstRef>,
        body: HashMap<InstRef, Inst<'s>>,
        body_order: Vec<InstRef>,
    },
    BlockValue(ConstOrInst<'s>),
    ContinueLoop(BlockRef),
    BreakLoop(BlockRef),
    Return(ConstOrInst<'s>),
    TmpLoopStateStorage {
        start: RegisterRef,
        continued: BTreeMap<BlockRef, RegisterRef>,
    },
    LoopStateStorage {
        start: ConstOrInst<'s>,
        continued: HashMap<InstRef, ConstOrInst<'s>>,
    },
}
impl<'s> Inst<'s> {
    pub fn dump_ordered(
        list: &HashMap<InstRef, Self>,
        order: &[InstRef],
        sink: &mut (impl std::io::Write + ?Sized),
    ) -> std::io::Result<()> {
        fn dump(
            ord: &InstRef,
            x: &Inst,
            sink: &mut (impl std::io::Write + ?Sized),
            depth: usize,
        ) -> std::io::Result<()> {
            let indent = IndentWriter(depth);

            match x {
                Inst::BuiltinIORef(b) => writeln!(sink, "{indent}{ord:?}: BuiltinIORef {b:?}"),
                Inst::DescriptorRef { set, binding } => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: DescriptorRef set={set} binding={binding}"
                    )
                }
                Inst::PushConstantRef(o) => {
                    writeln!(sink, "{indent}{ord:?}: PushConstantRef offset={o}")
                }
                Inst::WorkgroupSharedMemoryRef(rp) => {
                    writeln!(sink, "{indent}{ord:?}: WorkgroupSharedMemoryRef id={rp:?}")
                }
                Inst::CompositeRef(src, compo) => {
                    writeln!(sink, "{indent}{ord:?}: CompositeRef {src:?} {compo:?}")
                }
                Inst::IntrinsicBinary(left, op, right) => {
                    writeln!(sink, "{indent}{ord:?}: {left:?} `{op:?}` {right:?}")
                }
                Inst::IntrinsicUnary(value, op) => {
                    writeln!(sink, "{indent}{ord:?}: unary-{op:?} {value:?}")
                }
                Inst::Cast(src, t) => writeln!(sink, "{indent}{ord:?}: {src:?} as {t:?}"),
                Inst::ConstructIntrinsicType(it, xs) => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: {it:?}({:?})",
                        CommaSeparatedWriter(xs)
                    )
                }
                Inst::ConstructComposite(xs) => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: composite({:?})",
                        CommaSeparatedWriter(xs)
                    )
                }
                Inst::Swizzle(src, xs) => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: {src:?}.{}",
                        SwizzleElementWriter(xs)
                    )
                }
                Inst::CompositeInsert {
                    source,
                    index,
                    value,
                } => writeln!(
                    sink,
                    "{indent}{ord:?}: {source:?}.{} <- {value:?}",
                    SwizzleElementWriter(&[*index])
                ),
                Inst::LoadRef(ptr) => writeln!(sink, "{indent}{ord:?}: Load {ptr:?}"),
                Inst::StoreRef { ptr, value } => {
                    writeln!(sink, "{indent}{ord:?}: Store {ptr:?} <- {value:?}")
                }
                Inst::IntrinsicCall { identifier, args } => writeln!(
                    sink,
                    "{indent}{ord:?}: IntrinsicCall {identifier}({:?})",
                    CommaSeparatedWriter(args)
                ),
                Inst::Branch {
                    source,
                    true_instructions,
                    true_instruction_order,
                    false_instructions,
                    false_instruction_order,
                } => {
                    writeln!(sink, "{indent}{ord:?}: Branch if {source:?} {{")?;
                    for x in true_instruction_order.iter() {
                        dump(x, &true_instructions[x], sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}} else {{")?;
                    for x in false_instruction_order.iter() {
                        dump(x, &false_instructions[x], sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}}")
                }
                Inst::LoopWhile {
                    incoming_block,
                    condition,
                    condition_order,
                    body,
                    body_order,
                } => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: Loop (Enter @ {incoming_block}) while {{"
                    )?;
                    for x in condition_order.iter() {
                        dump(x, &condition[x], sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}} do {{")?;
                    for x in body_order.iter() {
                        dump(x, &body[x], sink, depth + 1)?;
                    }
                    writeln!(sink, "{indent}}}")
                }
                Inst::ContinueLoop(b) => writeln!(sink, "{indent}{ord:?}: continue @ {b}"),
                Inst::BreakLoop(b) => writeln!(sink, "{indent}{ord:?}: break @ {b}"),
                Inst::Return(v) => writeln!(sink, "{indent}{ord:?}: return {v:?}"),
                Inst::BlockValue(v) => writeln!(sink, "{indent}{ord:?}: block value = {v:?}"),
                Inst::TmpLoopStateStorage { start, continued } => {
                    writeln!(sink, "{indent}{ord:?}: (TmpLoopStateStorage) start={start:?} continued={continued:?}")
                }
                Inst::LoopStateStorage { start, continued } => {
                    writeln!(
                        sink,
                        "{indent}{ord:?}: LoopStateStorage start={start:?} continued={continued:?}"
                    )
                }
            }
        }

        for x in order.iter() {
            dump(x, &list[x], sink, 0)?;
        }

        Ok(())
    }

    fn flat_iter_mut(&mut self, mut process: impl FnMut(&mut Self) + Copy) {
        process(self);
        match self {
            Self::Branch {
                true_instructions,
                false_instructions,
                ..
            } => {
                for x in true_instructions
                    .iter_mut()
                    .chain(false_instructions.iter_mut())
                {
                    x.1.flat_iter_mut(process);
                }
            }
            Self::LoopWhile {
                condition, body, ..
            } => {
                for x in condition.iter_mut().chain(body.iter_mut()) {
                    x.1.flat_iter_mut(process);
                }
            }
            Self::BuiltinIORef(_)
            | Self::DescriptorRef { .. }
            | Self::PushConstantRef(_)
            | Self::WorkgroupSharedMemoryRef(_)
            | Self::CompositeRef(_, _)
            | Self::IntrinsicBinary(_, _, _)
            | Self::IntrinsicUnary(_, _)
            | Self::Cast(_, _)
            | Self::ConstructIntrinsicType(_, _)
            | Self::ConstructComposite(_)
            | Self::Swizzle(_, _)
            | Self::CompositeInsert { .. }
            | Self::LoadRef(_)
            | Self::StoreRef { .. }
            | Self::IntrinsicCall { .. }
            | Self::BlockValue(_)
            | Self::ContinueLoop(_)
            | Self::BreakLoop(_)
            | Self::Return(_)
            | Self::TmpLoopStateStorage { .. }
            | Self::LoopStateStorage { .. } => (),
        }
    }
}

pub struct Function<'s> {
    pub instructions: HashMap<InstRef, Inst<'s>>,
    pub instruction_order: Vec<InstRef>,
}

struct FnInstGenContext<'s> {
    last_ref_id: usize,
    generated_inst_register_map: HashMap<RegisterRef, ConstOrInst<'s>>,
    block_to_continue_inst: HashMap<BlockRef, InstRef>,
}
impl<'s> FnInstGenContext<'s> {
    fn new() -> Self {
        Self {
            last_ref_id: 0,
            generated_inst_register_map: HashMap::new(),
            block_to_continue_inst: HashMap::new(),
        }
    }

    fn alloc_inst_ref(&mut self) -> InstRef {
        self.last_ref_id += 1;

        InstRef(self.last_ref_id - 1)
    }

    fn map_register(&mut self, register: RegisterRef, ci: ConstOrInst<'s>) {
        self.generated_inst_register_map.insert(register, ci);
    }

    fn try_get_for_register<'c>(&'c self, register: &RegisterRef) -> Option<&'c ConstOrInst<'s>> {
        self.generated_inst_register_map.get(register)
    }

    fn has_generated(&self, register: &RegisterRef) -> bool {
        self.generated_inst_register_map.contains_key(register)
    }

    fn continue_inst_ref_for_block(&self, block: &BlockRef) -> InstRef {
        self.block_to_continue_inst
            .get(block)
            .copied()
            .expect("no continue executed in the block")
    }
}

struct LocalInstGenContext<'s> {
    pub instructions: HashMap<InstRef, Inst<'s>>,
    pub instruction_order: Vec<InstRef>,
}
impl<'s> LocalInstGenContext<'s> {
    fn new() -> Self {
        Self {
            instructions: HashMap::new(),
            instruction_order: Vec::new(),
        }
    }

    fn emit_inst(&mut self, fnctx: &mut FnInstGenContext<'s>, inst: Inst<'s>) -> InstRef {
        let ir = fnctx.alloc_inst_ref();
        self.instructions.insert(ir, inst);
        self.instruction_order.push(ir);

        ir
    }

    fn r#continue(&mut self, fnctx: &mut FnInstGenContext<'s>, execute_block: BlockRef) -> InstRef {
        let ir = self.emit_inst(fnctx, Inst::ContinueLoop(execute_block));
        fnctx.block_to_continue_inst.insert(execute_block, ir);

        ir
    }
}

/// Loop WhileのConditionの評価開始ブロック -> Loop Whileの実際のブロック を収集する
pub fn collect_condition_head_link(prg: &BlockifiedProgram) -> HashMap<BlockRef, BlockRef> {
    let mut map = HashMap::new();

    for (bx, b) in prg.iter_blocks_with_ref() {
        match b.flow {
            BlockFlowInstruction::ConditionalLoop { r#continue, .. } => {
                // 一旦ちゃんとつながるかとかは考えない(たぶん繋がってくれているはず)
                map.insert(r#continue, bx);
            }
            _ => (),
        }
    }

    map
}

/// IR2を構成する
pub fn reconstruct<'a, 's>(
    prg: &BlockifiedProgram<'a, 's>,
    chead_to_cloop: &HashMap<BlockRef, BlockRef>,
) -> Function<'s> {
    fn emit_const_or_inst<'a, 's>(
        prg: &BlockifiedProgram<'a, 's>,
        r: RegisterRef,
        fnctx: &mut FnInstGenContext<'s>,
        ctx: &mut LocalInstGenContext<'s>,
        impure_allowlist: &HashSet<usize>,
        loop_enter: Option<BlockRef>,
    ) -> ConstOrInst<'s> {
        if let Some(ir) = fnctx.try_get_for_register(&r) {
            // すでに生成済み
            return ir.clone();
        }

        match r {
            RegisterRef::Const(r) => match prg.constants[r].inst {
                BlockConstInstruction::BuiltinIORef(b) => {
                    let ir: ConstOrInst = ctx.emit_inst(fnctx, Inst::BuiltinIORef(b)).into();
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::PushConstantRef(o) => {
                    let ir: ConstOrInst = ctx.emit_inst(fnctx, Inst::PushConstantRef(o)).into();
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::DescriptorRef { set, binding } => {
                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::DescriptorRef { set, binding })
                        .into();
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::WorkgroupSharedMemoryRef(ref rp) => {
                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::WorkgroupSharedMemoryRef(rp.clone()))
                        .into();
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ScopeLocalVarRef(_, _) => {
                    unreachable!("scope local vars must be stripped")
                }
                BlockConstInstruction::FunctionInputVarRef(_, _) => {
                    unreachable!("function input vars must be stripped")
                }
                // そろそろこの辺を考え始めないといけない（usageをトラックして適切な型を推論するようにするのがよさそうか？）
                BlockConstInstruction::LitInt(ref l) => {
                    // 一旦IntはSIntで取り扱う（RustもC++もこれっぽいのでそれにならっておく）
                    let ir =
                        ConstOrInst::Const(ConstValue::SIntLit(ConstSIntLiteral(l.0.clone(), l.1)));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::LitNum(_) => todo!("litnum"),
                BlockConstInstruction::LitUInt(ref l) => {
                    let ir = ConstOrInst::Const(ConstValue::UIntLit(l.clone()));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::LitSInt(ref l) => {
                    let ir = ConstOrInst::Const(ConstValue::SIntLit(l.clone()));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::LitFloat(ref l) => {
                    let ir = ConstOrInst::Const(ConstValue::FloatLit(l.clone()));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmBool(true) => {
                    let ir = ConstOrInst::Const(ConstValue::True);
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmBool(false) => {
                    let ir = ConstOrInst::Const(ConstValue::False);
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmInt(v) => {
                    // 一旦IntはSIntで取り扱う（RustもC++もこれっぽいのでそれにならっておく）
                    let ir = ConstOrInst::Const(ConstValue::SInt(
                        v.try_into().expect("too large imm number"),
                    ));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmUInt(v) => {
                    let ir = ConstOrInst::Const(ConstValue::UInt(v));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmSInt(v) => {
                    let ir = ConstOrInst::Const(ConstValue::SInt(v));
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::ImmUnit => {
                    let ir = ConstOrInst::Const(ConstValue::Unit);
                    fnctx.map_register(RegisterRef::Const(r), ir.clone());
                    ir
                }
                BlockConstInstruction::IntrinsicFunctionRef(_) => {
                    unreachable!("unprocessed pseudo ref")
                }
                BlockConstInstruction::IntrinsicTypeConstructorRef(_) => {
                    unreachable!("unprocessed pseudo ref")
                }
                BlockConstInstruction::UserDefinedFunctionRef(_, _) => {
                    unreachable!("unprocessed udf ref")
                }
            },
            RegisterRef::Pure(r) => match prg.pure_instructions[r].inst {
                BlockPureInstruction::Cast(src, ref t) => {
                    let ConstOrInst::Inst(src) =
                        emit_const_or_inst(prg, src, fnctx, ctx, impure_allowlist, loop_enter)
                    else {
                        unreachable!("cannot emit const cast(needs precomputed)");
                    };

                    let ir: ConstOrInst = ctx.emit_inst(fnctx, Inst::Cast(src, t.clone())).into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::InstantiateIntrinsicTypeClass(_, _) => {
                    unreachable!("InstantiateIntrinsicTypeClass must be desugared");
                }
                BlockPureInstruction::PromoteIntToNumber(_) => {
                    unreachable!("PromoteIntToNumber must be desugared")
                }
                BlockPureInstruction::ConstructIntrinsicComposite(it, ref args) => {
                    let args = args
                        .iter()
                        .map(|a| {
                            emit_const_or_inst(prg, *a, fnctx, ctx, impure_allowlist, loop_enter)
                        })
                        .collect::<Vec<_>>();

                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::ConstructIntrinsicType(it, args))
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::ConstructTuple(ref args) => {
                    let args = args
                        .iter()
                        .map(|a| {
                            emit_const_or_inst(prg, *a, fnctx, ctx, impure_allowlist, loop_enter)
                        })
                        .collect::<Vec<_>>();

                    let ir: ConstOrInst =
                        ctx.emit_inst(fnctx, Inst::ConstructComposite(args)).into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::ConstructStruct(ref args) => {
                    let args = args
                        .iter()
                        .map(|a| {
                            emit_const_or_inst(prg, *a, fnctx, ctx, impure_allowlist, loop_enter)
                        })
                        .collect::<Vec<_>>();

                    let ir: ConstOrInst =
                        ctx.emit_inst(fnctx, Inst::ConstructComposite(args)).into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::ArrayRef { source, index } => {
                    let ConstOrInst::Inst(src) =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter)
                    else {
                        unreachable!("cannot CompositeRef for const value");
                    };
                    let index =
                        emit_const_or_inst(prg, index, fnctx, ctx, impure_allowlist, loop_enter);

                    let ir: ConstOrInst =
                        ctx.emit_inst(fnctx, Inst::CompositeRef(src, index)).into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::TupleRef(source, index) => {
                    let ConstOrInst::Inst(src) =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter)
                    else {
                        unreachable!("cannot CompositeRef for const value");
                    };

                    let ir: ConstOrInst = ctx
                        .emit_inst(
                            fnctx,
                            Inst::CompositeRef(
                                src,
                                ConstOrInst::Const(ConstValue::UInt(
                                    index.try_into().expect("too large tuple index"),
                                )),
                            ),
                        )
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::MemberRef(source, ref name) => {
                    let ConstOrInst::Inst(src) =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter)
                    else {
                        unreachable!("cannot CompositeRef for const value");
                    };
                    let member_index = match source
                        .ty(prg)
                        .as_dereferenced()
                        .expect("cannot dereference source of MemberRef")
                    {
                        ConcreteType::Struct(members) => members
                            .iter()
                            .position(|m| &m.name == name)
                            .expect("no member found"),
                        _ => unreachable!("cannot MemberRef to non-struct type"),
                    };

                    let ir: ConstOrInst = ctx
                        .emit_inst(
                            fnctx,
                            Inst::CompositeRef(
                                src,
                                ConstOrInst::Const(ConstValue::UInt(
                                    member_index.try_into().expect("too large struct"),
                                )),
                            ),
                        )
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::Swizzle(source, ref indices) => {
                    let source =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter);

                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::Swizzle(source, indices.clone()))
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::SwizzleRef(_, _) => {
                    unreachable!("SwizzleRef must be desugared")
                }
                BlockPureInstruction::IntrinsicBinaryOp(left, op, right) => {
                    let left =
                        emit_const_or_inst(prg, left, fnctx, ctx, impure_allowlist, loop_enter);
                    let right =
                        emit_const_or_inst(prg, right, fnctx, ctx, impure_allowlist, loop_enter);

                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::IntrinsicBinary(left, op, right))
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::IntrinsicUnaryOp(source, op) => {
                    let source =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter);

                    let ir: ConstOrInst = ctx
                        .emit_inst(fnctx, Inst::IntrinsicUnary(source, op))
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::CompositeInsert {
                    source,
                    value,
                    index,
                } => {
                    let source =
                        emit_const_or_inst(prg, source, fnctx, ctx, impure_allowlist, loop_enter);
                    let value =
                        emit_const_or_inst(prg, value, fnctx, ctx, impure_allowlist, loop_enter);

                    let ir: ConstOrInst = ctx
                        .emit_inst(
                            fnctx,
                            Inst::CompositeInsert {
                                source,
                                index,
                                value,
                            },
                        )
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
                BlockPureInstruction::StaticPathRef(_) => unreachable!("deprecated StaticPathRef"),
                BlockPureInstruction::RegisterAlias(_) => unreachable!("unresolved register alias"),
                BlockPureInstruction::PureFuncall(_, _) => unreachable!("non inlined funcall"),
                BlockPureInstruction::PureIntrinsicCall(id, ref args) => {
                    let args = args
                        .iter()
                        .map(|&a| {
                            emit_const_or_inst(prg, a, fnctx, ctx, impure_allowlist, loop_enter)
                        })
                        .collect::<Vec<_>>();

                    let ir: ConstOrInst = ctx
                        .emit_inst(
                            fnctx,
                            Inst::IntrinsicCall {
                                identifier: id,
                                args,
                            },
                        )
                        .into();
                    fnctx.map_register(RegisterRef::Pure(r), ir.clone());
                    ir
                }
            },
            // allowlistにあるImpure Register（同ブロック内）はここで処理してもいい
            RegisterRef::Impure(r) if impure_allowlist.contains(&r) => {
                match prg.impure_instructions[&r] {
                    BlockInstruction::LoadRef(ptr) => {
                        let ConstOrInst::Inst(ptr) =
                            emit_const_or_inst(prg, ptr, fnctx, ctx, impure_allowlist, loop_enter)
                        else {
                            unreachable!("cannot load from const");
                        };

                        let ir: ConstOrInst = ctx.emit_inst(fnctx, Inst::LoadRef(ptr)).into();
                        fnctx.map_register(RegisterRef::Impure(r), ir.clone());
                        ir
                    }
                    BlockInstruction::Phi(ref incomings) => {
                        // 一時的にそのままもってくる(あとで変換する)
                        let loop_enter = loop_enter.expect("phi appeared out of loop condition");
                        let mut incomings = incomings.clone();
                        let start_reg = incomings.remove(&loop_enter).expect("no enter state?");
                        let ir: ConstOrInst = ctx
                            .emit_inst(
                                fnctx,
                                Inst::TmpLoopStateStorage {
                                    start: start_reg,
                                    continued: incomings,
                                },
                            )
                            .into();
                        fnctx.map_register(RegisterRef::Impure(r), ir.clone());
                        ir
                    }
                }
            }
            RegisterRef::Impure(r) => todo!("impure to const_or_inst: r{r}"),
        }
    }

    fn process<'a, 's>(
        prg: &BlockifiedProgram<'a, 's>,
        n: crate::ir::block::BlockRef,
        until: Option<crate::ir::block::BlockRef>,
        last_block: &mut crate::ir::block::BlockRef,
        chead_to_cloop: &HashMap<BlockRef, BlockRef>,
        allow_condition_head: bool,
        flow_only_process: bool,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
        fnctx: &mut FnInstGenContext<'s>,
        ctx: &mut LocalInstGenContext<'s>,
    ) {
        eprintln!("process: {n:?} {flow_only_process}");
        if !flow_only_process {
            if until.is_some_and(|x| x == n) {
                // ここまで
                return;
            }

            if !allow_condition_head {
                if let Some(&clp) = chead_to_cloop.get(&n) {
                    // このブロックはCondition Head: Loop Whileの解釈にまかせる
                    eprintln!("chead jmp {n:?} -> {clp:?}");
                    process(
                        prg,
                        clp,
                        until,
                        last_block,
                        chead_to_cloop,
                        false,
                        true,
                        loop_stack,
                        fnctx,
                        ctx,
                    );
                    return;
                }
            }

            for imr in prg.blocks[n.0].eval_impure_registers.iter() {
                if fnctx.has_generated(&RegisterRef::Impure(*imr)) {
                    // すでに生成済み
                    continue;
                }

                match prg.impure_instructions[imr] {
                    BlockInstruction::LoadRef(ptr) => {
                        let ConstOrInst::Inst(ptr) = emit_const_or_inst(
                            prg,
                            ptr,
                            fnctx,
                            ctx,
                            &prg.blocks[n.0].eval_impure_registers,
                            None,
                        ) else {
                            unreachable!("cannot load from const");
                        };

                        let ir: ConstOrInst = ctx.emit_inst(fnctx, Inst::LoadRef(ptr)).into();
                        fnctx.map_register(RegisterRef::Impure(*imr), ir);
                    }
                    BlockInstruction::Phi(ref _incomings) => {
                        // 一時的にそのままもってくる(あとで変換する)
                        // let ir: ConstOrInst = ctx
                        //     .emit_inst(fnctx, Inst::TmpLoopStateStorage(incomings.clone()))
                        //     .into();
                        // fnctx.map_register(RegisterRef::Impure(*imr), ir);
                        unimplemented!("phi appeared outside of condition?");
                    }
                }
            }
        }

        let incoming_block = *last_block;
        *last_block = n;

        match prg.blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                if !flow_only_process {
                    // 次の実行ブロックにPhiがあったらそのレジスタをここで評価する
                    for ir in prg.blocks[next.0].eval_impure_registers.iter() {
                        match prg.impure_instructions[ir] {
                            BlockInstruction::Phi(ref incomings) => {
                                if let Some(&r) = incomings.get(&n) {
                                    emit_const_or_inst(
                                        prg,
                                        r,
                                        fnctx,
                                        ctx,
                                        &prg.blocks[n.0].eval_impure_registers,
                                        None,
                                    );
                                }
                            }
                            _ => (),
                        }
                    }
                }

                process(
                    prg,
                    next,
                    until,
                    last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    ctx,
                );
            }
            BlockFlowInstruction::StoreRef { ptr, value, after } => {
                let ConstOrInst::Inst(ptr) =
                    emit_const_or_inst(prg, ptr, fnctx, ctx, &HashSet::new(), None)
                else {
                    unreachable!("cannot store to const value");
                };
                let value = emit_const_or_inst(prg, value, fnctx, ctx, &HashSet::new(), None);
                ctx.emit_inst(fnctx, Inst::StoreRef { ptr, value });

                if !flow_only_process {
                    // 次の実行ブロックにPhiがあったらそのレジスタをここで評価する
                    for ir in prg.blocks[after.unwrap().0].eval_impure_registers.iter() {
                        match prg.impure_instructions[ir] {
                            BlockInstruction::Phi(ref incomings) => {
                                if let Some(&r) = incomings.get(&n) {
                                    emit_const_or_inst(
                                        prg,
                                        r,
                                        fnctx,
                                        ctx,
                                        &prg.blocks[n.0].eval_impure_registers,
                                        None,
                                    );
                                }
                            }
                            _ => (),
                        }
                    }
                }

                process(
                    prg,
                    after.unwrap(),
                    until,
                    last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    ctx,
                );
            }
            BlockFlowInstruction::Funcall {
                callee,
                ref args,
                result,
                after_return,
            } => {
                unimplemented!("FunctionCall {callee:?}");
            }
            BlockFlowInstruction::IntrinsicImpureFuncall {
                identifier,
                ref args,
                result,
                after_return,
            } => {
                let arg_refs = args
                    .iter()
                    .map(|&a| emit_const_or_inst(prg, a, fnctx, ctx, &HashSet::new(), None))
                    .collect::<Vec<_>>();

                let ir: ConstOrInst = ctx
                    .emit_inst(
                        fnctx,
                        Inst::IntrinsicCall {
                            identifier,
                            args: arg_refs,
                        },
                    )
                    .into();
                fnctx.map_register(result, ir);

                if !flow_only_process {
                    // 次の実行ブロックにPhiがあったらそのレジスタをここで評価する
                    for ir in prg.blocks[after_return.unwrap().0]
                        .eval_impure_registers
                        .iter()
                    {
                        match prg.impure_instructions[ir] {
                            BlockInstruction::Phi(ref incomings) => {
                                if let Some(&r) = incomings.get(&n) {
                                    emit_const_or_inst(
                                        prg,
                                        r,
                                        fnctx,
                                        ctx,
                                        &prg.blocks[n.0].eval_impure_registers,
                                        None,
                                    );
                                }
                            }
                            _ => (),
                        }
                    }
                }

                process(
                    prg,
                    after_return.unwrap(),
                    until,
                    last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    ctx,
                );
            }
            BlockFlowInstruction::Conditional {
                source,
                r#true,
                r#false,
                merge,
            } => {
                let source = emit_const_or_inst(prg, source, fnctx, ctx, &HashSet::new(), None);

                let mut true_local_ctx = LocalInstGenContext::new();
                let mut true_last_block = n;
                process(
                    prg,
                    r#true,
                    Some(merge),
                    &mut true_last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    &mut true_local_ctx,
                );
                let mut false_local_ctx = LocalInstGenContext::new();
                let mut false_last_block = n;
                process(
                    prg,
                    r#false,
                    Some(merge),
                    &mut false_last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    &mut false_local_ctx,
                );

                if let Some((merge_phi_register, merge_phi_incomings)) = prg.blocks[merge.0]
                    .eval_impure_registers
                    .iter()
                    .find_map(|r| match prg.impure_instructions[r] {
                        BlockInstruction::Phi(ref incomings) => Some((*r, incomings)),
                        _ => None,
                    })
                {
                    let Some(&true_phi_incoming_register) =
                        merge_phi_incomings.get(&true_last_block)
                    else {
                        unreachable!(
                            "no true phi incoming register true_last_blk=b{}",
                            true_last_block.0
                        );
                    };
                    let Some(&false_phi_incoming_register) =
                        merge_phi_incomings.get(&false_last_block)
                    else {
                        unreachable!(
                            "no false phi incoming register false_last_blk=b{}",
                            false_last_block.0
                        );
                    };

                    let true_phi_value = emit_const_or_inst(
                        prg,
                        true_phi_incoming_register,
                        fnctx,
                        &mut true_local_ctx,
                        &HashSet::new(),
                        None,
                    );
                    let false_phi_value = emit_const_or_inst(
                        prg,
                        false_phi_incoming_register,
                        fnctx,
                        &mut false_local_ctx,
                        &HashSet::new(),
                        None,
                    );

                    true_local_ctx.emit_inst(fnctx, Inst::BlockValue(true_phi_value.clone()));
                    false_local_ctx.emit_inst(fnctx, Inst::BlockValue(false_phi_value.clone()));
                    let ir: ConstOrInst = ctx
                        .emit_inst(
                            fnctx,
                            Inst::Branch {
                                source,
                                true_instructions: true_local_ctx.instructions,
                                true_instruction_order: true_local_ctx.instruction_order,
                                false_instructions: false_local_ctx.instructions,
                                false_instruction_order: false_local_ctx.instruction_order,
                            },
                        )
                        .into();
                    fnctx.map_register(RegisterRef::Impure(merge_phi_register), ir);
                } else {
                    ctx.emit_inst(
                        fnctx,
                        Inst::Branch {
                            source,
                            true_instructions: true_local_ctx.instructions,
                            true_instruction_order: true_local_ctx.instruction_order,
                            false_instructions: false_local_ctx.instructions,
                            false_instruction_order: false_local_ctx.instruction_order,
                        },
                    );
                }

                process(
                    prg,
                    merge,
                    until,
                    last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    ctx,
                );
            }
            BlockFlowInstruction::ConditionalEnd => (/* end */),
            BlockFlowInstruction::ConditionalLoop {
                condition,
                r#break,
                r#continue,
                body,
            } => {
                let mut condition_local_ctx = LocalInstGenContext::new();
                let mut condition_last_block = r#continue;
                process(
                    prg,
                    r#continue,
                    Some(n),
                    &mut condition_last_block,
                    chead_to_cloop,
                    true,
                    false,
                    loop_stack,
                    fnctx,
                    &mut condition_local_ctx,
                );

                // eval this block(last part of condition)
                for imr in prg.blocks[n.0].eval_impure_registers.iter() {
                    if fnctx.has_generated(&RegisterRef::Impure(*imr)) {
                        // すでに生成済み
                        continue;
                    }

                    match prg.impure_instructions[imr] {
                        BlockInstruction::LoadRef(ptr) => {
                            let ConstOrInst::Inst(ptr) = emit_const_or_inst(
                                prg,
                                ptr,
                                fnctx,
                                &mut condition_local_ctx,
                                &prg.blocks[n.0].eval_impure_registers,
                                Some(incoming_block),
                            ) else {
                                unreachable!("cannot load from const");
                            };

                            let ir: ConstOrInst = condition_local_ctx
                                .emit_inst(fnctx, Inst::LoadRef(ptr))
                                .into();
                            fnctx.map_register(RegisterRef::Impure(*imr), ir);
                        }
                        BlockInstruction::Phi(ref incomings) => {
                            // unimplemented!("phi decoding: {incomings:?}")
                            let mut incomings = incomings.clone();
                            let start_reg =
                                incomings.remove(&incoming_block).expect("no enter state?");
                            let ir: ConstOrInst = condition_local_ctx
                                .emit_inst(
                                    fnctx,
                                    Inst::TmpLoopStateStorage {
                                        start: start_reg,
                                        continued: incomings,
                                    },
                                )
                                .into();
                            fnctx.map_register(RegisterRef::Impure(*imr), ir);
                        }
                    }
                }

                eprintln!("cont end");
                let condition = emit_const_or_inst(
                    prg,
                    condition,
                    fnctx,
                    &mut condition_local_ctx,
                    &prg.blocks[n.0].eval_impure_registers,
                    Some(incoming_block),
                );
                condition_local_ctx.emit_inst(fnctx, Inst::BlockValue(condition));

                let mut body_local_ctx = LocalInstGenContext::new();
                let mut body_last_block = body;
                loop_stack.push((r#break, r#continue));
                process(
                    prg,
                    body,
                    Some(r#break),
                    &mut body_last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    &mut body_local_ctx,
                );
                loop_stack.pop();

                ctx.emit_inst(
                    fnctx,
                    Inst::LoopWhile {
                        incoming_block,
                        condition: condition_local_ctx.instructions,
                        condition_order: condition_local_ctx.instruction_order,
                        body: body_local_ctx.instructions,
                        body_order: body_local_ctx.instruction_order,
                    },
                );
                process(
                    prg,
                    r#break,
                    until,
                    last_block,
                    chead_to_cloop,
                    false,
                    false,
                    loop_stack,
                    fnctx,
                    ctx,
                );
            }
            BlockFlowInstruction::Break => {
                let &(b, _) = loop_stack.last().unwrap();

                if !flow_only_process {
                    // 次の実行ブロックにPhiがあったらそのレジスタをここで評価する
                    for ir in prg.blocks[b.0].eval_impure_registers.iter() {
                        match prg.impure_instructions[ir] {
                            BlockInstruction::Phi(ref incomings) => {
                                if let Some(&r) = incomings.get(&n) {
                                    emit_const_or_inst(
                                        prg,
                                        r,
                                        fnctx,
                                        ctx,
                                        &prg.blocks[n.0].eval_impure_registers,
                                        None,
                                    );
                                }
                            }
                            _ => (),
                        }
                    }
                }

                ctx.emit_inst(fnctx, Inst::BreakLoop(n));
            }
            BlockFlowInstruction::Continue => {
                let &(_, c) = loop_stack.last().unwrap();

                if !flow_only_process {
                    // 次の実行ブロックにPhiがあったらそのレジスタをここで評価する
                    for ir in prg.blocks[c.0].eval_impure_registers.iter() {
                        match prg.impure_instructions[ir] {
                            BlockInstruction::Phi(ref incomings) => {
                                if let Some(&r) = incomings.get(&n) {
                                    emit_const_or_inst(
                                        prg,
                                        r,
                                        fnctx,
                                        ctx,
                                        &prg.blocks[n.0].eval_impure_registers,
                                        None,
                                    );
                                }
                            }
                            _ => (),
                        }
                    }
                }

                ctx.r#continue(fnctx, n);
            }
            BlockFlowInstruction::Return(v) => {
                let v = emit_const_or_inst(prg, v, fnctx, ctx, &HashSet::new(), None);
                ctx.emit_inst(fnctx, Inst::Return(v));
            }
            BlockFlowInstruction::Undetermined => unreachable!("undetermined destination"),
        }
    }

    let mut function_ctx = FnInstGenContext::new();
    let mut function_local_ctx = LocalInstGenContext::new();

    process(
        prg,
        crate::ir::block::BlockRef(0),
        None,
        &mut crate::ir::block::BlockRef(0),
        chead_to_cloop,
        false,
        false,
        &mut Vec::new(),
        &mut function_ctx,
        &mut function_local_ctx,
    );

    // TmpLoopStateStorageをLoopStateStorageに変換する（仮置きしてたRegisterの消化）
    for x in function_local_ctx.instructions.iter_mut() {
        x.1.flat_iter_mut(|x| match x {
            Inst::TmpLoopStateStorage {
                start,
                ref continued,
            } => {
                let new_start = function_ctx
                    .try_get_for_register(&start)
                    .expect("no inst generated for start state register")
                    .clone();
                let new_continued = continued
                    .iter()
                    .map(|(b, r)| {
                        (
                            function_ctx.continue_inst_ref_for_block(b),
                            function_ctx
                                .try_get_for_register(r)
                                .expect("no inst generated for register")
                                .clone(),
                        )
                    })
                    .collect();

                *x = Inst::LoopStateStorage {
                    start: new_start,
                    continued: new_continued,
                };
            }
            _ => (),
        });
    }

    Function {
        instructions: function_local_ctx.instructions,
        instruction_order: function_local_ctx.instruction_order,
    }
}
