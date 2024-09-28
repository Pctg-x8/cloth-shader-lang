use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    parser::StatementNode,
    ref_path::RefPath,
    scope::{SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    spirv as spv,
    symbol::{meta::BuiltinInputOutput, IntrinsicFunctionSymbol},
    utils::PtrEq,
};

use super::{
    expr::{binary_op, simplify_expression, simplify_lefthand_expression, ConstModifiers},
    Const, ConstFloatLiteral, ConstIntLiteral, ConstNumberLiteral, ConstSIntLiteral,
    ConstUIntLiteral, ExprRef,
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegisterRef(pub usize);
impl RegisterRef {
    pub fn ty<'c, 's>(
        self,
        p: &'c (impl RegisterTypeProvider<'c, 's> + ?Sized),
    ) -> &'c ConcreteType<'s> {
        p.register_type(self)
    }
}

pub trait RegisterTypeProvider<'c, 's> {
    fn register_type(&'c self, register: RegisterRef) -> &'c ConcreteType<'s>;
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockRef(pub usize);

#[derive(Debug, Clone)]
pub enum BlockFlowInstruction {
    Goto(BlockRef),
    Conditional {
        source: RegisterRef,
        r#true: BlockRef,
        r#false: BlockRef,
        merge: BlockRef,
    },
    Funcall {
        callee: RegisterRef,
        args: Vec<RegisterRef>,
        result: RegisterRef,
        after_return: Option<BlockRef>,
    },
    IntrinsicImpureFuncall {
        identifier: &'static str,
        args: Vec<RegisterRef>,
        result: RegisterRef,
        after_return: Option<BlockRef>,
    },
    StoreRef {
        ptr: RegisterRef,
        value: RegisterRef,
        after: Option<BlockRef>,
    },
    Return(RegisterRef),
    ConditionalLoop {
        condition: RegisterRef,
        r#break: BlockRef,
        r#continue: BlockRef,
        body: BlockRef,
    },
    Break,
    Continue,
    Undetermined,
}
impl BlockFlowInstruction {
    pub fn relocate_result_register(
        &mut self,
        mut relocator: impl FnMut(&mut RegisterRef),
    ) -> bool {
        match self {
            Self::Funcall { ref mut result, .. }
            | Self::IntrinsicImpureFuncall { ref mut result, .. } => {
                let x0 = result.0;
                relocator(result);
                result.0 != x0
            }
            Self::Goto(_)
            | Self::StoreRef { .. }
            | Self::Conditional { .. }
            | Self::ConditionalLoop { .. }
            | Self::Break
            | Self::Continue
            | Self::Return(_)
            | Self::Undetermined => false,
        }
    }

    pub fn relocate_register(&mut self, mut relocator: impl FnMut(&mut RegisterRef)) -> bool {
        match self {
            Self::Goto(_) => false,
            Self::StoreRef {
                ref mut ptr,
                ref mut value,
                ..
            } => {
                let x0 = value.0;
                relocator(value);
                let p0 = ptr.0;
                relocator(ptr);
                value.0 != x0 || ptr.0 != p0
            }
            Self::Funcall {
                ref mut callee,
                ref mut args,
                ..
            } => {
                let x0 = callee.0;
                relocator(callee);
                let a = args.iter_mut().fold(false, |a, r| {
                    let x0 = r.0;
                    relocator(r);
                    r.0 != x0 || a
                });
                callee.0 != x0 || a
            }
            Self::IntrinsicImpureFuncall { ref mut args, .. } => {
                args.iter_mut().fold(false, |a, r| {
                    let x0 = r.0;
                    relocator(r);
                    r.0 != x0 || a
                })
            }
            Self::Conditional { ref mut source, .. } => {
                let x0 = source.0;
                relocator(source);
                source.0 != x0
            }
            Self::ConditionalLoop {
                ref mut condition, ..
            } => {
                let x0 = condition.0;
                relocator(condition);
                condition.0 != x0
            }
            Self::Break | Self::Continue => false,
            Self::Return(ref mut r) => {
                let x0 = r.0;
                relocator(r);
                r.0 != x0
            }
            Self::Undetermined => false,
        }
    }

    pub fn relocate_next_block(&mut self, mut relocator: impl FnMut(&mut BlockRef)) -> bool {
        match self {
            Self::Goto(ref mut next)
            | Self::StoreRef {
                after: Some(ref mut next),
                ..
            }
            | Self::Funcall {
                after_return: Some(ref mut next),
                ..
            }
            | Self::IntrinsicImpureFuncall {
                after_return: Some(ref mut next),
                ..
            } => {
                let n0 = next.0;
                relocator(next);
                next.0 != n0
            }
            Self::Conditional {
                ref mut r#true,
                ref mut r#false,
                ref mut merge,
                ..
            } => {
                let (t0, f0, m0) = (r#true.0, r#false.0, merge.0);
                relocator(r#true);
                relocator(r#false);
                relocator(merge);
                r#true.0 != t0 || r#false.0 != f0 || merge.0 != m0
            }
            Self::ConditionalLoop {
                ref mut r#break,
                ref mut r#continue,
                ref mut body,
                ..
            } => {
                let (b0, c0, bd0) = (r#break.0, r#continue.0, body.0);
                relocator(r#break);
                relocator(r#continue);
                relocator(body);
                r#break.0 != b0 || r#continue.0 != c0 || body.0 != bd0
            }
            BlockFlowInstruction::StoreRef { after: None, .. }
            | BlockFlowInstruction::Funcall {
                after_return: None, ..
            }
            | BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: None, ..
            }
            | BlockFlowInstruction::Break
            | BlockFlowInstruction::Continue
            | BlockFlowInstruction::Undetermined
            | BlockFlowInstruction::Return(_) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LogAnd,
    LogOr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicUnaryOperation {
    Neg,
    BitNot,
    LogNot,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockInstruction<'a, 's> {
    Cast(RegisterRef, ConcreteType<'s>),
    InstantiateIntrinsicTypeClass(RegisterRef, IntrinsicType),
    LoadRef(RegisterRef),
    ConstructIntrinsicComposite(IntrinsicType, Vec<RegisterRef>),
    ConstructTuple(Vec<RegisterRef>),
    ConstructStruct(Vec<RegisterRef>),
    PromoteIntToNumber(RegisterRef),
    IntrinsicBinaryOp(RegisterRef, IntrinsicBinaryOperation, RegisterRef),
    IntrinsicUnaryOp(RegisterRef, IntrinsicUnaryOperation),
    ConstUnit,
    IntrinsicFunctionRef(Vec<IntrinsicFunctionSymbol>),
    IntrinsicTypeConstructorRef(IntrinsicType),
    ScopeLocalVarRef(PtrEq<'a, SymbolScope<'a, 's>>, usize),
    FunctionInputVarRef(PtrEq<'a, SymbolScope<'a, 's>>, usize),
    UserDefinedFunctionRef(PtrEq<'a, SymbolScope<'a, 's>>, SourceRefSliceEq<'s>),
    BuiltinIORef(BuiltinInputOutput),
    DescriptorRef {
        set: u32,
        binding: u32,
    },
    PushConstantRef(u32),
    Swizzle(RegisterRef, Vec<usize>),
    SwizzleRef(RegisterRef, Vec<usize>),
    MemberRef(RegisterRef, SourceRefSliceEq<'s>),
    WorkgroupSharedMemoryRef(RefPath),
    StaticPathRef(RefPath),
    ArrayRef {
        source: RegisterRef,
        index: RegisterRef,
    },
    TupleRef(RegisterRef, usize),
    ConstInt(ConstIntLiteral<'s>),
    ConstUInt(ConstUIntLiteral<'s>),
    ConstSInt(ConstSIntLiteral<'s>),
    ConstNumber(ConstNumberLiteral<'s>),
    ConstFloat(ConstFloatLiteral<'s>),
    ImmBool(bool),
    ImmInt(isize),
    ImmSInt(i32),
    ImmUInt(u32),
    Phi(BTreeMap<BlockRef, RegisterRef>),
    RegisterAlias(RegisterRef),
    PureIntrinsicCall(&'static str, Vec<RegisterRef>),
    PureFuncall(RegisterRef, Vec<RegisterRef>),
    CompositeInsert {
        value: RegisterRef,
        source: RegisterRef,
        index: usize,
    },
}
impl<'a, 's> BlockInstruction<'a, 's> {
    #[inline(always)]
    pub const fn is_block_dependent(&self) -> bool {
        match self {
            Self::Phi { .. } => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub const fn is_const(&self) -> bool {
        matches!(
            self,
            Self::ConstInt(_)
                | Self::ConstUInt(_)
                | Self::ConstSInt(_)
                | Self::ConstNumber(_)
                | Self::ConstFloat(_)
                | Self::ConstUnit
                | Self::ImmBool(_)
                | Self::ImmInt(_)
                | Self::ImmSInt(_)
                | Self::ImmUInt(_)
                | Self::ScopeLocalVarRef(_, _)
                | Self::FunctionInputVarRef(_, _)
                | Self::UserDefinedFunctionRef(_, _)
                | Self::IntrinsicFunctionRef(_)
                | Self::IntrinsicTypeConstructorRef(_)
        )
    }

    #[inline(always)]
    pub fn dup_phi_incoming(&mut self, old: BlockRef, new: BlockRef) {
        if let Self::Phi(ref mut incomings) = self {
            if let Some(&r) = incomings.get(&old) {
                incomings.insert(new, r);
            }
        }
    }

    #[inline(always)]
    pub fn try_instantiate_const(&self) -> Option<Const> {
        match self {
            Self::ConstInt(v) => Some(Const::Int(v.instantiate())),
            Self::ConstSInt(v) => Some(Const::SInt(v.instantiate())),
            Self::ConstUInt(v) => Some(Const::UInt(v.instantiate())),
            &Self::ImmBool(v) => Some(Const::Bool(v)),
            &Self::ImmInt(v) => Some(Const::Int(v)),
            &Self::ImmSInt(v) => Some(Const::SInt(v)),
            &Self::ImmUInt(v) => Some(Const::UInt(v)),
            _ => None,
        }
    }

    pub fn relocate_block_ref(&mut self, mut relocator: impl FnMut(&mut BlockRef)) -> bool {
        match self {
            Self::Phi(ref mut incomings) => {
                let mut modified = false;
                for (mut k, v) in core::mem::replace(incomings, BTreeMap::new()) {
                    let k0 = k.0;
                    relocator(&mut k);
                    modified = modified || k.0 != k0;
                    assert!(incomings.insert(k, v).is_none());
                }
                modified
            }
            _ => false,
        }
    }

    #[inline(always)]
    pub fn apply_register_alias(&mut self, map: &HashMap<RegisterRef, RegisterRef>) -> bool {
        self.relocate_register(|r| {
            while let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    pub fn relocate_register(&mut self, mut relocator: impl FnMut(&mut RegisterRef)) -> bool {
        match self {
            Self::ConstUnit
            | Self::ConstInt(_)
            | Self::ConstNumber(_)
            | Self::ConstSInt(_)
            | Self::ConstUInt(_)
            | Self::ConstFloat(_)
            | Self::ImmBool(_)
            | Self::ImmInt(_)
            | Self::ImmSInt(_)
            | Self::ImmUInt(_)
            | Self::ScopeLocalVarRef(_, _)
            | Self::FunctionInputVarRef(_, _)
            | Self::UserDefinedFunctionRef(_, _)
            | Self::IntrinsicFunctionRef(_)
            | Self::IntrinsicTypeConstructorRef(_)
            | Self::BuiltinIORef(_)
            | Self::DescriptorRef { .. }
            | Self::PushConstantRef(_)
            | Self::StaticPathRef(_)
            | Self::WorkgroupSharedMemoryRef(_) => false,
            Self::Cast(ref mut x, _)
            | Self::InstantiateIntrinsicTypeClass(ref mut x, _)
            | Self::RegisterAlias(ref mut x)
            | Self::IntrinsicUnaryOp(ref mut x, _)
            | Self::LoadRef(ref mut x)
            | Self::PromoteIntToNumber(ref mut x)
            | Self::Swizzle(ref mut x, _)
            | Self::SwizzleRef(ref mut x, _)
            | Self::MemberRef(ref mut x, _)
            | Self::TupleRef(ref mut x, _) => {
                let x0 = x.0;
                relocator(x);
                x.0 != x0
            }
            Self::IntrinsicBinaryOp(ref mut x, _, ref mut y)
            | Self::ArrayRef {
                source: ref mut x,
                index: ref mut y,
            } => {
                let (x0, y0) = (x.0, y.0);
                relocator(x);
                relocator(y);
                x.0 != x0 || y.0 != y0
            }
            Self::ConstructIntrinsicComposite(_, ref mut xs)
            | Self::ConstructTuple(ref mut xs)
            | Self::ConstructStruct(ref mut xs)
            | Self::PureIntrinsicCall(_, ref mut xs)
            | Self::PureFuncall(_, ref mut xs) => xs.iter_mut().fold(false, |modified, x| {
                let x0 = x.0;
                relocator(x);
                modified || x.0 != x0
            }),
            Self::Phi(ref mut incomings) => incomings.values_mut().fold(false, |modified, x| {
                let x0 = x.0;
                relocator(x);
                modified || x.0 != x0
            }),
            Self::CompositeInsert {
                ref mut value,
                ref mut source,
                ..
            } => {
                let (x0, y0) = (value.0, source.0);
                relocator(value);
                relocator(source);
                x0 != value.0 || y0 != source.0
            }
        }
    }

    pub fn dump(&self, w: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::Cast(RegisterRef(src), ty) => write!(w, "r{src} as {ty:?}"),
            Self::InstantiateIntrinsicTypeClass(RegisterRef(src), ty) => {
                write!(w, "Instantiate(r{src} as {ty:?})")
            }
            Self::PromoteIntToNumber(RegisterRef(value)) => {
                write!(w, "PromoteIntToNumber(r{value})")
            }
            Self::ConstInt(ConstIntLiteral(repr, modifiers)) => {
                write!(w, "ConstInt({repr:?}, mod={modifiers:?})")
            }
            Self::ConstNumber(ConstNumberLiteral(repr, modifiers)) => {
                write!(w, "ConstNumber({repr:?}, mod={modifiers:?})")
            }
            Self::ConstUInt(ConstUIntLiteral(repr, modifiers)) => {
                write!(w, "ConstUInt({repr:?}, mod={modifiers:?})")
            }
            Self::ConstSInt(ConstSIntLiteral(repr, modifiers)) => {
                write!(w, "ConstSInt({repr:?}, mod={modifiers:?})")
            }
            Self::ConstFloat(ConstFloatLiteral(repr, modifiers)) => {
                write!(w, "ConstFloat({repr:?}, mod={modifiers:?})")
            }
            Self::ImmBool(value) => write!(w, "ImmBool {value}"),
            Self::ImmInt(value) => write!(w, "ImmInt {value}"),
            Self::ImmSInt(value) => write!(w, "ImmSInt {value}"),
            Self::ImmUInt(value) => write!(w, "ImmUInt {value}"),
            Self::ConstUnit => write!(w, "()"),
            Self::LoadRef(RegisterRef(ptr)) => write!(w, "Load r{ptr}"),
            Self::ScopeLocalVarRef(scope, var_id) => {
                write!(w, "ScopeLocalVarRef({var_id}) in {scope:?}")
            }
            Self::FunctionInputVarRef(scope, var_id) => {
                write!(w, "FunctionInputVarRef({var_id}) in {scope:?}")
            }
            Self::BuiltinIORef(b) => write!(w, "BuiltinIORef {b:?}"),
            Self::DescriptorRef { set, binding } => {
                write!(w, "DescriptorRef set={set}, binding={binding}")
            }
            Self::PushConstantRef(offset) => write!(w, "PushConstantRef offset={offset}"),
            Self::MemberRef(RegisterRef(source), name) => write!(w, "ref r{source}.({name:?})"),
            Self::WorkgroupSharedMemoryRef(path) => write!(w, "ref[WorkgroupSharedMem] {path:?}"),
            Self::StaticPathRef(path) => write!(w, "ref {path:?}"),
            Self::ArrayRef {
                source: RegisterRef(source),
                index: RegisterRef(index),
            } => write!(w, "ref r{source}[r{index}]"),
            Self::TupleRef(RegisterRef(source), index) => write!(w, "ref r{source}.{index}"),
            Self::IntrinsicFunctionRef(overloads) => write!(
                w,
                "IntrinsicFunctionRef<{}>",
                overloads
                    .iter()
                    .map(|x| format!("{x:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::IntrinsicTypeConstructorRef(ty) => {
                write!(w, "IntrinsicTypeConstructorRef<{ty:?}>")
            }
            Self::UserDefinedFunctionRef(scope, name) => {
                write!(w, "UserDefinedFunctionRef({name:?}) in {scope:?}")
            }
            Self::SwizzleRef(RegisterRef(source), elements) => write!(
                w,
                "ref r{source}.{}",
                elements
                    .iter()
                    .map(|x| ['x', 'y', 'z', 'w'][*x])
                    .collect::<String>()
            ),
            Self::Swizzle(RegisterRef(source), elements) => write!(
                w,
                "r{source}.{}",
                elements
                    .iter()
                    .map(|x| ['x', 'y', 'z', 'w'][*x])
                    .collect::<String>()
            ),
            Self::Phi(incoming_selectors) => write!(
                w,
                "phi [{}]",
                incoming_selectors
                    .iter()
                    .map(|(BlockRef(from), RegisterRef(r))| format!("{from} -> r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ConstructIntrinsicComposite(it, args) => write!(
                w,
                "ConstructIntrinsicComposite#{it:?}({})",
                args.iter()
                    .map(|RegisterRef(r)| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ConstructTuple(values) => write!(
                w,
                "({})",
                values
                    .iter()
                    .map(|RegisterRef(r)| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ConstructStruct(values) => write!(
                w,
                "{{ {} }}",
                values
                    .iter()
                    .map(|RegisterRef(r)| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::IntrinsicBinaryOp(RegisterRef(left), op, RegisterRef(right)) => {
                write!(w, "{op:?}(r{left}, r{right})")
            }
            Self::IntrinsicUnaryOp(RegisterRef(value), op) => write!(w, "{op:?} r{value}"),
            Self::RegisterAlias(RegisterRef(source)) => write!(w, "r{source}"),
            Self::PureIntrinsicCall(identifier, args) => write!(
                w,
                "PureIntrinsicCall {identifier}({})",
                args.iter()
                    .map(|RegisterRef(r)| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::PureFuncall(RegisterRef(source), args) => write!(
                w,
                "PureFuncall r{source}({})",
                args.iter()
                    .map(|RegisterRef(r)| format!("r{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::CompositeInsert {
                value: RegisterRef(value),
                source: RegisterRef(source),
                index,
            } => write!(w, "CompositeInsert r{source}.{index} <- r{value}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub eval_registers: HashSet<RegisterRef>,
    pub flow: BlockFlowInstruction,
}
impl Block {
    #[inline(always)]
    pub fn flow_only(flow: BlockFlowInstruction) -> Self {
        Self {
            eval_registers: HashSet::new(),
            flow,
        }
    }

    #[inline(always)]
    pub fn has_block_dependent_instructions(
        &self,
        mod_instructions: &HashMap<RegisterRef, BlockInstruction<'_, '_>>,
    ) -> bool {
        self.eval_registers.iter().any(|r| {
            mod_instructions
                .get(r)
                .is_some_and(|x| x.is_block_dependent())
        })
    }

    #[inline(always)]
    pub fn is_loop_term_block(&self) -> bool {
        matches!(
            self.flow,
            BlockFlowInstruction::Break | BlockFlowInstruction::Continue
        )
    }

    #[inline(always)]
    pub fn apply_flow_register_alias(&mut self, map: &HashMap<RegisterRef, RegisterRef>) -> bool {
        self.flow.relocate_register(|r| {
            while let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    #[inline(always)]
    pub fn relocate_flow_register(
        &mut self,
        relocator: impl FnMut(&mut RegisterRef) + Copy,
    ) -> bool {
        self.flow.relocate_register(relocator)
    }

    #[inline(always)]
    pub fn relocate_register(
        &mut self,
        mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'_, '_>>,
        relocator: impl FnMut(&mut RegisterRef) + Copy,
    ) -> bool {
        let mut mod_insts = false;
        for r in self.eval_registers.iter() {
            let modified = match mod_instructions.get_mut(r) {
                Some(x) => x.relocate_register(relocator),
                None => false,
            };
            mod_insts = mod_insts || modified;
        }
        let mod_flow = self.flow.relocate_register(relocator);

        mod_insts || mod_flow
    }

    /// phiにoldからのものがあったらnewからのものにコピー
    #[inline]
    pub fn dup_phi_incoming(
        &mut self,
        mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'_, '_>>,
        old: BlockRef,
        new: BlockRef,
    ) {
        for r in self.eval_registers.iter() {
            if let Some(x) = mod_instructions.get_mut(r) {
                x.dup_phi_incoming(old, new);
            }
        }
    }

    pub fn try_set_next(&mut self, next: BlockRef) -> bool {
        match self.flow {
            BlockFlowInstruction::Undetermined => {
                self.flow = BlockFlowInstruction::Goto(next);
                true
            }
            BlockFlowInstruction::Funcall {
                ref mut after_return,
                ..
            } if after_return.is_none() => {
                *after_return = Some(next);
                true
            }
            BlockFlowInstruction::StoreRef { ref mut after, .. } if after.is_none() => {
                *after = Some(next);
                true
            }
            _ => false,
        }
    }

    pub fn try_set_branch(
        &mut self,
        condition: RegisterRef,
        r#true: BlockRef,
        r#false: BlockRef,
        merge: BlockRef,
    ) -> bool {
        match self.flow {
            BlockFlowInstruction::Undetermined => {
                self.flow = BlockFlowInstruction::Conditional {
                    source: condition,
                    r#true,
                    r#false,
                    merge,
                };
                true
            }
            _ => false,
        }
    }
}

pub struct BlockInstructionEmitter<'c, 'a, 's> {
    pub generation_context: &'c mut BlockGenerationContext<'a, 's>,
    pub instruction_emission_context: &'c mut BlockInstructionEmissionContext<'a, 's>,
    pub block_eval_registers: HashSet<RegisterRef>,
}
impl<'c, 'a, 's> RegisterTypeProvider<'c, 's> for BlockInstructionEmitter<'c, 'a, 's> {
    #[inline(always)]
    fn register_type(&'c self, register: RegisterRef) -> &'c ConcreteType<'s> {
        self.instruction_emission_context.register_type(register)
    }
}
impl<'c, 'a, 's> BlockInstructionEmitter<'c, 'a, 's> {
    #[inline]
    pub fn new(
        generation_context: &'c mut BlockGenerationContext<'a, 's>,
        instruction_emission_context: &'c mut BlockInstructionEmissionContext<'a, 's>,
    ) -> Self {
        Self {
            generation_context,
            instruction_emission_context,
            block_eval_registers: HashSet::new(),
        }
    }

    #[inline(always)]
    pub fn into_block(self, flow: BlockFlowInstruction) -> Block {
        Block {
            eval_registers: self.block_eval_registers,
            flow,
        }
    }

    #[inline(always)]
    pub fn create_block(self, flow: BlockFlowInstruction) -> BlockRef {
        let blk = Block {
            eval_registers: self.block_eval_registers,
            flow,
        };

        self.generation_context.add(blk)
    }

    pub fn into_eval_registers(self) -> HashSet<RegisterRef> {
        self.block_eval_registers
    }

    fn add_instruction(&mut self, dest_reg: RegisterRef, inst: BlockInstruction<'a, 's>) {
        self.instruction_emission_context
            .instructions
            .insert(dest_reg, inst);
        self.block_eval_registers.insert(dest_reg);
    }

    #[inline]
    pub fn loaded(&mut self, ptr: RegisterRef) -> RegisterRef {
        match ptr.ty(self) {
            ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => self.load_ref(ptr),
            _ => ptr,
        }
    }

    pub fn cast(&mut self, src: RegisterRef, to: ConcreteType<'s>) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(to.clone());
        self.add_instruction(dest_register, BlockInstruction::Cast(src, to));

        dest_register
    }

    pub fn instantiate_intrinsic_type_class(
        &mut self,
        src: RegisterRef,
        to: IntrinsicType,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(to.into());
        self.add_instruction(
            dest_register,
            BlockInstruction::InstantiateIntrinsicTypeClass(src, to),
        );

        dest_register
    }

    pub fn load_ref(&mut self, ptr: RegisterRef) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(ptr.ty(self).as_dereferenced().unwrap().clone());
        self.add_instruction(dest_register, BlockInstruction::LoadRef(ptr));

        dest_register
    }

    pub fn construct_intrinsic_composite(
        &mut self,
        it: IntrinsicType,
        args: Vec<RegisterRef>,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(it.into());
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstructIntrinsicComposite(it, args),
        );

        dest_register
    }

    pub fn construct_tuple(&mut self, elements: Vec<RegisterRef>) -> RegisterRef {
        let ty = ConcreteType::Tuple(elements.iter().map(|r| r.ty(self).clone()).collect());

        let dest_register = self.instruction_emission_context.alloc_register(ty);

        dest_register
    }

    pub fn construct_struct(
        &mut self,
        elements: Vec<RegisterRef>,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(out_type);
        self.add_instruction(dest_register, BlockInstruction::ConstructStruct(elements));

        dest_register
    }

    pub fn promote_int_to_number(&mut self, r: RegisterRef) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(ConcreteType::UnknownNumberClass);
        self.add_instruction(dest_register, BlockInstruction::PromoteIntToNumber(r));

        dest_register
    }

    pub fn intrinsic_binary_op(
        &mut self,
        left: RegisterRef,
        op: IntrinsicBinaryOperation,
        right: RegisterRef,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(out_type);
        self.add_instruction(
            dest_register,
            BlockInstruction::IntrinsicBinaryOp(left, op, right),
        );

        dest_register
    }

    pub fn intrinsic_unary_op(
        &mut self,
        value: RegisterRef,
        op: IntrinsicUnaryOperation,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(out_type);
        self.add_instruction(dest_register, BlockInstruction::IntrinsicUnaryOp(value, op));

        dest_register
    }

    pub fn intrinsic_function_ref(
        &mut self,
        overloads: Vec<IntrinsicFunctionSymbol>,
    ) -> RegisterRef {
        let dest_register =
            self.instruction_emission_context
                .alloc_register(ConcreteType::OverloadedFunctions(
                    overloads
                        .iter()
                        .map(|s| (s.args.clone(), Box::new(s.output.clone())))
                        .collect(),
                ));
        self.add_instruction(
            dest_register,
            BlockInstruction::IntrinsicFunctionRef(overloads),
        );

        dest_register
    }

    pub fn intrinsic_type_constructor_ref(&mut self, it: IntrinsicType) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(ConcreteType::IntrinsicTypeConstructor(it));
        self.add_instruction(
            dest_register,
            BlockInstruction::IntrinsicTypeConstructorRef(it),
        );

        dest_register
    }

    pub fn scope_local_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::ScopeLocalVarRef(PtrEq(scope), var_id),
        );

        dest_register
    }

    pub fn scope_local_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::ScopeLocalVarRef(PtrEq(scope), var_id),
        );

        dest_register
    }

    pub fn function_input_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::FunctionInputVarRef(PtrEq(scope), var_id),
        );

        dest_register
    }

    pub fn function_input_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::FunctionInputVarRef(PtrEq(scope), var_id),
        );

        dest_register
    }

    pub fn user_defined_function_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        name: SourceRef<'s>,
    ) -> RegisterRef {
        let fs = scope.user_defined_function_symbol(name.slice).unwrap();
        let ty = ConcreteType::Function {
            args: fs.inputs.iter().map(|(_, _, _, t)| t.clone()).collect(),
            output: match fs.output.len() {
                0 => None,
                1 => Some(Box::new(fs.output[0].1.clone())),
                _ => Some(Box::new(ConcreteType::Tuple(
                    fs.output.iter().map(|(_, t)| t.clone()).collect(),
                ))),
            },
        };

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::UserDefinedFunctionRef(PtrEq(scope), SourceRefSliceEq(name)),
        );

        dest_register
    }

    pub fn swizzle(&mut self, source: RegisterRef, elements: Vec<usize>) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self)
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        );

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::Swizzle(source, elements));

        dest_register
    }

    pub fn swizzle_ref(&mut self, source: RegisterRef, elements: Vec<usize>) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self)
                .as_dereferenced()
                .unwrap()
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        )
        .imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::SwizzleRef(source, elements),
        );

        dest_register
    }

    pub fn swizzle_mutable_ref(
        &mut self,
        source: RegisterRef,
        elements: Vec<usize>,
    ) -> RegisterRef {
        let ty = ConcreteType::from(
            source
                .ty(self)
                .as_dereferenced()
                .unwrap()
                .scalar_type()
                .unwrap()
                .of_vector(elements.len().try_into().unwrap())
                .unwrap(),
        )
        .mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::SwizzleRef(source, elements),
        );

        dest_register
    }

    pub fn member_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = member_type.imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::MemberRef(source, SourceRefSliceEq(name)),
        );

        dest_register
    }

    pub fn member_mutable_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = member_type.mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(
            dest_register,
            BlockInstruction::MemberRef(source, SourceRefSliceEq(name)),
        );

        dest_register
    }

    pub fn array_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::ArrayRef { source, index });

        dest_register
    }

    pub fn array_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::ArrayRef { source, index });

        dest_register
    }

    pub fn tuple_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.imm_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::TupleRef(source, index));

        dest_register
    }

    pub fn tuple_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        let ty = element_type.mutable_ref();

        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::TupleRef(source, index));

        dest_register
    }

    pub fn const_int(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(ConcreteType::UnknownIntClass);
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstInt(ConstIntLiteral(
                SourceRefSliceEq(repr),
                ConstModifiers::empty(),
            )),
        );

        dest_register
    }

    pub fn const_number(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(ConcreteType::UnknownNumberClass);
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstNumber(ConstNumberLiteral(
                SourceRefSliceEq(repr),
                ConstModifiers::empty(),
            )),
        );

        dest_register
    }

    pub fn const_uint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(IntrinsicType::UInt.into());
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstUInt(ConstUIntLiteral(
                SourceRefSliceEq(repr),
                ConstModifiers::empty(),
            )),
        );

        dest_register
    }

    pub fn const_sint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(IntrinsicType::SInt.into());
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstSInt(ConstSIntLiteral(
                SourceRefSliceEq(repr),
                ConstModifiers::empty(),
            )),
        );

        dest_register
    }

    pub fn const_float(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .alloc_register(IntrinsicType::Float.into());
        self.add_instruction(
            dest_register,
            BlockInstruction::ConstFloat(ConstFloatLiteral(
                SourceRefSliceEq(repr),
                ConstModifiers::empty(),
            )),
        );

        dest_register
    }

    pub fn phi(
        &mut self,
        incoming_selectors: BTreeMap<BlockRef, RegisterRef>,
        ty: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self.instruction_emission_context.alloc_register(ty);
        self.add_instruction(dest_register, BlockInstruction::Phi(incoming_selectors));

        dest_register
    }
}

pub struct BlockInstructionEmissionContext<'a, 's> {
    pub registers: Vec<ConcreteType<'s>>,
    pub instructions: HashMap<RegisterRef, BlockInstruction<'a, 's>>,
}
impl<'c, 'a, 's> RegisterTypeProvider<'c, 's> for BlockInstructionEmissionContext<'a, 's> {
    fn register_type(&'c self, register: RegisterRef) -> &'c ConcreteType<'s> {
        &self.registers[register.0]
    }
}
impl<'a, 's> BlockInstructionEmissionContext<'a, 's> {
    pub fn new() -> Self {
        Self {
            registers: Vec::new(),
            instructions: HashMap::new(),
        }
    }

    pub fn alloc_register(&mut self, ty: ConcreteType<'s>) -> RegisterRef {
        self.registers.push(ty);

        RegisterRef(self.registers.len() - 1)
    }

    #[inline]
    pub fn const_unit(&mut self) -> RegisterRef {
        let r = self.alloc_register(IntrinsicType::Unit.into());
        self.instructions.insert(r, BlockInstruction::ConstUnit);

        r
    }
}

pub struct BlockGenerationContext<'a, 's> {
    pub symbol_scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    pub blocks: Vec<Block>,
}
impl<'a, 's> BlockGenerationContext<'a, 's> {
    #[inline]
    pub fn new(symbol_scope_arena: &'a Arena<SymbolScope<'a, 's>>) -> Self {
        Self {
            symbol_scope_arena,
            blocks: Vec::new(),
        }
    }

    pub fn add(&mut self, blk: Block) -> BlockRef {
        self.blocks.push(blk);

        BlockRef(self.blocks.len() - 1)
    }

    #[inline(always)]
    pub fn block_mut(&mut self, index: BlockRef) -> &mut Block {
        &mut self.blocks[index.0]
    }

    #[inline(always)]
    pub fn try_chain(&mut self, from: BlockRef, to: BlockRef) -> bool {
        self.blocks[from.0].try_set_next(to)
    }

    pub fn dump_blocks(
        &self,
        writer: &mut (impl Write + ?Sized),
        registers: &[ConcreteType<'s>],
        mod_instructions: &HashMap<RegisterRef, BlockInstruction>,
    ) -> std::io::Result<()> {
        writeln!(writer, "Registers: ")?;
        dump_registers(writer, registers)?;
        dump_blocks(writer, &self.blocks, mod_instructions)?;

        Ok(())
    }
}

pub fn dump_registers(
    writer: &mut (impl std::io::Write + ?Sized),
    registers: &[ConcreteType],
) -> std::io::Result<()> {
    for (n, r) in registers.iter().enumerate() {
        writeln!(writer, "  r{n}: {r:?}")?;
    }

    Ok(())
}

pub fn dump_blocks(
    writer: &mut (impl std::io::Write + ?Sized),
    blocks: &[Block],
    mod_instructions: &HashMap<RegisterRef, BlockInstruction>,
) -> std::io::Result<()> {
    for (n, b) in blocks.iter().enumerate() {
        writeln!(writer, "b{n}: {{")?;

        if !b.eval_registers.is_empty() {
            let mut sorted = b.eval_registers.iter().collect::<Vec<_>>();
            sorted.sort_by_key(|&RegisterRef(r)| r);
            for r in sorted {
                if let Some(x) = mod_instructions.get(&r) {
                    write!(writer, "  r{} = ", r.0)?;
                    x.dump(writer)?;
                    writer.write(b"\n")?;
                }
            }

            writer.write(b"\n")?;
        }

        match b.flow {
            BlockFlowInstruction::Goto(BlockRef(x)) => writeln!(writer, "  goto -> b{x}")?,
            BlockFlowInstruction::StoreRef {
                ptr: RegisterRef(ptr),
                value: RegisterRef(value),
                after: Some(BlockRef(after)),
            } => writeln!(writer, "  *r{ptr} = r{value} -> b{after}")?,
            BlockFlowInstruction::StoreRef {
                ptr: RegisterRef(ptr),
                value: RegisterRef(value),
                after: None,
            } => writeln!(writer, "  *r{ptr} = r{value} -> ???")?,
            BlockFlowInstruction::Funcall {
                callee: RegisterRef(callee),
                ref args,
                result: RegisterRef(result),
                after_return: Some(BlockRef(after_return)),
            } => writeln!(
                writer,
                "  r{result} = (r{callee})({}) -> b{after_return}",
                args.iter()
                    .map(|r| format!("r{}", r.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?,
            BlockFlowInstruction::Funcall {
                callee: RegisterRef(callee),
                ref args,
                result: RegisterRef(result),
                after_return: None,
            } => writeln!(
                writer,
                "  r{result} = (r{callee})({}) -> ???",
                args.iter()
                    .map(|r| format!("r{}", r.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?,
            BlockFlowInstruction::IntrinsicImpureFuncall { identifier, ref args, result: RegisterRef(result), after_return: Some(BlockRef(after_return)) } => writeln!(
                writer,
                "  r{result} = IntrinsicImpureFuncall {identifier}({}) -> b{after_return}",
                args.iter().map(|r| format!("r{}", r.0)).collect::<Vec<_>>().join(", ")
            )?,
            BlockFlowInstruction::IntrinsicImpureFuncall { identifier, ref args, result: RegisterRef(result), after_return: None } => writeln!(
                writer,
                "  r{result} = IntrinsicImpureFuncall {identifier}({}) -> ???",
                args.iter().map(|r| format!("r{}", r.0)).collect::<Vec<_>>().join(", ")
            )?,
            BlockFlowInstruction::Conditional {
                source: RegisterRef(source),
                r#true: BlockRef(t),
                r#false: BlockRef(e),
                merge: BlockRef(merge),
            } => writeln!(
                writer,
                "  branch r{source} ? -> b{t} : -> b{e} merge at b{merge}"
            )?,
            BlockFlowInstruction::Return(RegisterRef(r)) => writeln!(writer, "  return r{r}")?,
            BlockFlowInstruction::ConditionalLoop {
                condition: RegisterRef(condition),
                r#break: BlockRef(brk),
                r#continue: BlockRef(cont),
                body: BlockRef(body),
            } => writeln!(
                writer,
                "  loop while r{condition} ... break -> b{brk}, continue -> b{cont}, body -> b{body}"
            )?,
            BlockFlowInstruction::Break => writeln!(writer, "  break")?,
            BlockFlowInstruction::Continue => writeln!(writer, "  continue")?,
            BlockFlowInstruction::Undetermined => writeln!(writer, "  -> ???")?,
        }

        writeln!(writer, "}}")?;
    }

    Ok(())
}

pub fn transform_statement<'a, 's>(
    statement: StatementNode<'s>,
    scope: &'a SymbolScope<'a, 's>,
    ctx: &mut BlockGenerationContext<'a, 's>,
    inst_ctx: &mut BlockInstructionEmissionContext<'a, 's>,
) -> (BlockRef, BlockRef) {
    match statement {
        StatementNode::Let {
            mut_token,
            varname_token,
            expr,
            ..
        } => {
            let expr_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let expr = simplify_expression(expr, ctx, inst_ctx, expr_scope);

            let mut inst = BlockInstructionEmitter::new(ctx, inst_ctx);
            let expr_value = inst.loaded(expr.result);
            let VarId::ScopeLocal(var_id) = scope.declare_local_var(
                SourceRef::from(&varname_token),
                expr_value.ty(inst.instruction_emission_context).clone(),
                mut_token.is_some(),
            ) else {
                unreachable!();
            };
            let mutable_ref = inst.scope_local_var_mutable_ref(scope, var_id);
            let init_blk = inst.create_block(BlockFlowInstruction::StoreRef {
                ptr: mutable_ref,
                value: expr_value,
                after: None,
            });

            assert!(
                ctx.try_chain(expr.end_block, init_blk),
                "expr multiple out?"
            );

            (expr.start_block, init_blk)
        }
        StatementNode::OpEq {
            left_expr,
            opeq_token,
            expr,
        } => {
            let left_ref_expr = simplify_lefthand_expression(left_expr, ctx, inst_ctx, scope);
            let right_expr_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let right_expr = simplify_expression(expr, ctx, inst_ctx, right_expr_scope);

            let mut inst = BlockInstructionEmitter::new(ctx, inst_ctx);
            let right_value = inst.loaded(right_expr.result);
            let right_value = match opeq_token.slice {
                "=" => right_value,
                "+=" | "-=" | "*=" | "/=" | "%=" | "^^=" | "&=" | "|=" | "^=" | "<<=" | ">>="
                | "&&=" | "||=" => {
                    let left_expr_loaded =
                        match left_ref_expr.result.ty(inst.instruction_emission_context) {
                            ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => {
                                inst.load_ref(left_ref_expr.result)
                            }
                            _ => unreachable!("cannot store to right hand value"),
                        };

                    binary_op(
                        left_expr_loaded,
                        SourceRef {
                            slice: &opeq_token.slice[..opeq_token.slice.len() - 1],
                            line: opeq_token.line,
                            col: opeq_token.col,
                        },
                        right_value,
                        &mut inst,
                    )
                }
                _ => unreachable!("unknown opeq token"),
            };
            let right_value = match left_ref_expr.result.ty(inst.instruction_emission_context) {
                ConcreteType::MutableRef(inner) => {
                    match (right_value.ty(inst.instruction_emission_context), &**inner) {
                        (a, b) if a == b => right_value,
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::SInt),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::SInt)
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::UInt),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::UInt)
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::Float)
                        }
                        (
                            ConcreteType::UnknownNumberClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            inst.instantiate_intrinsic_type_class(right_value, IntrinsicType::Float)
                        }
                        (res_ty, dest_ty) => {
                            panic!("Error: cannot assign: src={res_ty:?}, dest={dest_ty:?}")
                        }
                    }
                }
                ConcreteType::Ref(t) => {
                    panic!(
                        "Error: cannot assign to immutable variable: {:?}",
                        t.clone().imm_ref()
                    )
                }
                _ => panic!("Error: cannot assign to non-ref type value"),
            };

            let perform_block = inst.create_block(BlockFlowInstruction::StoreRef {
                ptr: left_ref_expr.result,
                value: right_value,
                after: None,
            });
            assert!(
                ctx.try_chain(left_ref_expr.end_block, right_expr.start_block),
                "left multiple out?"
            );
            assert!(
                ctx.try_chain(right_expr.end_block, perform_block),
                "right multiple out?"
            );

            (left_ref_expr.start_block, perform_block)
        }
        StatementNode::Expression(x) => {
            let expr = simplify_expression(x, ctx, inst_ctx, scope);

            (expr.start_block, expr.end_block)
        }
        StatementNode::While {
            condition, runs, ..
        } => {
            let condition_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let condition = simplify_expression(condition, ctx, inst_ctx, condition_scope);
            let runs_scope = ctx.symbol_scope_arena.alloc(scope.new_child());
            let runs = simplify_expression(runs, ctx, inst_ctx, runs_scope);

            let runs_final_block = ctx.add(Block::flow_only(BlockFlowInstruction::Continue));
            let merge_block = ctx.add(Block::flow_only(BlockFlowInstruction::Undetermined));
            let loop_head_block =
                ctx.add(Block::flow_only(BlockFlowInstruction::ConditionalLoop {
                    condition: condition.result,
                    r#break: merge_block,
                    r#continue: condition.start_block,
                    body: runs.start_block,
                }));

            assert!(
                ctx.try_chain(condition.end_block, loop_head_block),
                "condition multiple out?"
            );
            assert!(
                ctx.try_chain(runs.end_block, runs_final_block),
                "runs multiple out?"
            );

            (condition.start_block, merge_block)
        }
    }
}

pub fn parse_incoming_flows(blocks: &[Block]) -> HashMap<BlockRef, Vec<BlockRef>> {
    let mut incomings = HashMap::new();
    let mut processed = HashSet::new();
    let mut loop_stack = Vec::new();

    fn parse(
        blocks: &[Block],
        n: BlockRef,
        incomings: &mut HashMap<BlockRef, Vec<BlockRef>>,
        processed: &mut HashSet<BlockRef>,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
    ) {
        if processed.contains(&n) {
            return;
        }

        processed.insert(n);
        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                incomings.entry(next).or_insert_with(Vec::new).push(n);
                parse(blocks, next, incomings, processed, loop_stack);
            }
            BlockFlowInstruction::StoreRef {
                after: Some(after), ..
            } => {
                incomings.entry(after).or_insert_with(Vec::new).push(n);
                parse(blocks, after, incomings, processed, loop_stack);
            }
            BlockFlowInstruction::StoreRef { .. } => (),
            BlockFlowInstruction::Funcall {
                after_return: Some(after_return),
                ..
            } => {
                incomings
                    .entry(after_return)
                    .or_insert_with(Vec::new)
                    .push(n);
                parse(blocks, after_return, incomings, processed, loop_stack);
            }
            BlockFlowInstruction::Funcall { .. } => (),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(after_return),
                ..
            } => {
                incomings
                    .entry(after_return)
                    .or_insert_with(Vec::new)
                    .push(n);
                parse(blocks, after_return, incomings, processed, loop_stack);
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                incomings.entry(r#true).or_insert_with(Vec::new).push(n);
                incomings.entry(r#false).or_insert_with(Vec::new).push(n);
                parse(blocks, r#true, incomings, processed, loop_stack);
                parse(blocks, r#false, incomings, processed, loop_stack);
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                incomings.entry(body).or_insert_with(Vec::new).push(n);
                incomings.entry(r#break).or_insert_with(Vec::new).push(n);

                loop_stack.push((r#break, r#continue));
                parse(blocks, body, incomings, processed, loop_stack);
                parse(blocks, r#break, incomings, processed, loop_stack);
                loop_stack.pop();
            }
            BlockFlowInstruction::Break => {
                let &(brk, _) = loop_stack.last().unwrap();
                incomings.entry(brk).or_insert_with(Vec::new).push(n);
            }
            BlockFlowInstruction::Continue => {
                let &(_, cont) = loop_stack.last().unwrap();
                incomings.entry(cont).or_insert_with(Vec::new).push(n);
            }
            BlockFlowInstruction::Return(_) => (),
            BlockFlowInstruction::Undetermined => (),
        }
    }

    parse(
        blocks,
        BlockRef(0),
        &mut incomings,
        &mut processed,
        &mut loop_stack,
    );
    incomings
}
