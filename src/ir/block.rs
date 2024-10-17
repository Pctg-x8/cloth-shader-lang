use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
    ops::RangeInclusive,
};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    parser::StatementNode,
    ref_path::RefPath,
    scope::{SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    symbol::{meta::BuiltinInputOutput, IntrinsicFunctionSymbol},
    utils::PtrEq,
};

use super::{
    expr::{binary_op, simplify_expression, simplify_lefthand_expression},
    ConstFloatLiteral, ConstIntLiteral, ConstModifiers, ConstNumberLiteral, ConstSIntLiteral,
    ConstUIntLiteral, LosslessConst,
};

pub struct BaseRegisters {
    pub r#const: usize,
    pub pure: usize,
    pub impure: usize,
}

pub enum ConcreteTypeRef<'c, 's> {
    Computed(ConcreteType<'s>),
    Ref(&'c ConcreteType<'s>),
}
impl core::fmt::Debug for ConcreteTypeRef<'_, '_> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Computed(x) => x.fmt(f),
            Self::Ref(x) => x.fmt(f),
        }
    }
}
impl<'s> core::ops::Deref for ConcreteTypeRef<'_, 's> {
    type Target = ConcreteType<'s>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Computed(r) => r,
            Self::Ref(r) => *r,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegisterRef {
    Const(usize),
    Pure(usize),
    Impure(usize),
}
impl RegisterRef {
    #[inline(always)]
    pub fn ty<'c, 's>(
        self,
        p: &'c (impl RegisterTypeProvider<'c, 's> + ?Sized),
    ) -> &'c ConcreteType<'s> {
        p.register_type(self)
    }

    #[inline(always)]
    pub const fn based_on(self, base: &BaseRegisters) -> Self {
        match self {
            Self::Const(n) => Self::Const(n + base.r#const),
            Self::Pure(n) => Self::Pure(n + base.pure),
            Self::Impure(n) => Self::Impure(n + base.impure),
        }
    }

    #[inline]
    pub fn entropy_order(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a.cmp(b),
            (Self::Const(_), Self::Pure(_)) => core::cmp::Ordering::Less,
            (Self::Const(_), Self::Impure(_)) => core::cmp::Ordering::Less,
            (Self::Pure(_), Self::Const(_)) => core::cmp::Ordering::Greater,
            (Self::Pure(a), Self::Pure(b)) => a.cmp(b),
            (Self::Pure(_), Self::Impure(_)) => core::cmp::Ordering::Less,
            (Self::Impure(_), Self::Const(_)) => core::cmp::Ordering::Greater,
            (Self::Impure(_), Self::Pure(_)) => core::cmp::Ordering::Greater,
            (Self::Impure(a), Self::Impure(b)) => a.cmp(b),
        }
    }

    #[inline]
    pub const fn as_id(self) -> usize {
        match self {
            Self::Const(x) | Self::Pure(x) | Self::Impure(x) => x,
        }
    }
}
impl core::fmt::Display for RegisterRef {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const(x) => write!(f, "c{x}"),
            Self::Pure(x) => write!(f, "x{x}"),
            Self::Impure(x) => write!(f, "r{x}"),
        }
    }
}

pub type ImpureInstructionMap = HashMap<usize, BlockInstruction>;
pub type PureInstructions<'s> = Vec<TypedBlockPureInstruction<'s>>;
pub type Constants<'a, 's> = Vec<TypedBlockConstInstruction<'a, 's>>;

pub type RegisterAliasMap = HashMap<RegisterRef, RegisterRef>;

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
    ConditionalEnd,
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
                let x0 = *result;
                relocator(result);
                *result != x0
            }
            Self::Goto(_)
            | Self::StoreRef { .. }
            | Self::Conditional { .. }
            | Self::ConditionalLoop { .. }
            | Self::Break
            | Self::Continue
            | Self::Return(_)
            | Self::Undetermined
            | Self::ConditionalEnd => false,
        }
    }

    #[inline(always)]
    pub fn apply_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.relocate_register(|r| {
            while let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    #[inline(always)]
    pub fn apply_parallel_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.relocate_register(|r| {
            if let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    pub fn enumerate_ref_registers(&self, mut reporter: impl FnMut(RegisterRef)) {
        match self {
            Self::Goto(_) => (),
            &Self::StoreRef { ptr, value, .. } => {
                reporter(ptr);
                reporter(value);
            }
            &Self::Funcall {
                callee, ref args, ..
            } => {
                reporter(callee);
                for x in args {
                    reporter(*x);
                }
            }
            &Self::IntrinsicImpureFuncall { ref args, .. } => {
                for x in args {
                    reporter(*x);
                }
            }
            &Self::Conditional { source, .. } => reporter(source),
            &Self::ConditionalLoop { condition, .. } => reporter(condition),
            Self::Break | Self::Continue => (),
            &Self::Return(r) => reporter(r),
            Self::Undetermined | Self::ConditionalEnd => (),
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
                let x0 = *value;
                relocator(value);
                let p0 = *ptr;
                relocator(ptr);
                *value != x0 || *ptr != p0
            }
            Self::Funcall {
                ref mut callee,
                ref mut args,
                ..
            } => {
                let x0 = *callee;
                relocator(callee);
                let a = args.iter_mut().fold(false, |a, r| {
                    let x0 = *r;
                    relocator(r);
                    *r != x0 || a
                });
                *callee != x0 || a
            }
            Self::IntrinsicImpureFuncall { ref mut args, .. } => {
                args.iter_mut().fold(false, |a, r| {
                    let x0 = *r;
                    relocator(r);
                    *r != x0 || a
                })
            }
            Self::Conditional { ref mut source, .. } => {
                let x0 = *source;
                relocator(source);
                *source != x0
            }
            Self::ConditionalLoop {
                ref mut condition, ..
            } => {
                let x0 = *condition;
                relocator(condition);
                *condition != x0
            }
            Self::Break | Self::Continue => false,
            Self::Return(ref mut r) => {
                let x0 = *r;
                relocator(r);
                *r != x0
            }
            Self::Undetermined | Self::ConditionalEnd => false,
        }
    }

    pub fn enumerate_dest_register(&self, mut reporter: impl FnMut(RegisterRef)) {
        match self {
            Self::Goto(_)
            | Self::StoreRef { .. }
            | Self::Conditional { .. }
            | Self::ConditionalLoop { .. }
            | Self::Break
            | Self::Continue
            | Self::Return(_)
            | Self::Undetermined
            | Self::ConditionalEnd => (),
            &Self::Funcall { result, .. } | &Self::IntrinsicImpureFuncall { result, .. } => {
                reporter(result);
            }
        }
    }

    #[inline]
    pub fn relocate_dest_register(&mut self, mut relocator: impl FnMut(&mut RegisterRef)) -> bool {
        match self {
            Self::Goto(_)
            | Self::StoreRef { .. }
            | Self::Conditional { .. }
            | Self::ConditionalLoop { .. }
            | Self::Break
            | Self::Continue
            | Self::Return(_)
            | Self::Undetermined
            | Self::ConditionalEnd => false,
            Self::Funcall { ref mut result, .. }
            | Self::IntrinsicImpureFuncall { ref mut result, .. } => {
                let x0 = *result;
                relocator(result);
                x0 != *result
            }
        }
    }

    pub fn relocate_block_ref(&mut self, mut relocator: impl FnMut(&mut BlockRef)) -> bool {
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
            Self::StoreRef { after: None, .. }
            | Self::Funcall {
                after_return: None, ..
            }
            | Self::IntrinsicImpureFuncall {
                after_return: None, ..
            }
            | Self::Break
            | Self::Continue
            | Self::Undetermined
            | Self::Return(_)
            | Self::ConditionalEnd => false,
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
pub enum BlockConstInstruction<'a, 's> {
    LitInt(ConstIntLiteral<'s>),
    LitNum(ConstNumberLiteral<'s>),
    LitUInt(ConstUIntLiteral<'s>),
    LitSInt(ConstSIntLiteral<'s>),
    LitFloat(ConstFloatLiteral<'s>),
    ImmUnit,
    ImmBool(bool),
    ImmInt(isize),
    ImmUInt(u32),
    ImmSInt(i32),
    IntrinsicFunctionRef(Vec<IntrinsicFunctionSymbol>),
    IntrinsicTypeConstructorRef(IntrinsicType),
    ScopeLocalVarRef(PtrEq<'a, SymbolScope<'a, 's>>, usize),
    FunctionInputVarRef(PtrEq<'a, SymbolScope<'a, 's>>, usize),
    UserDefinedFunctionRef(PtrEq<'a, SymbolScope<'a, 's>>, SourceRefSliceEq<'s>),
    BuiltinIORef(BuiltinInputOutput),
    DescriptorRef { set: u32, binding: u32 },
    PushConstantRef(u32),
    WorkgroupSharedMemoryRef(RefPath),
}
impl<'a, 's> BlockConstInstruction<'a, 's> {
    #[inline(always)]
    pub fn try_instantiate_lossless_const(&self) -> Option<LosslessConst> {
        match self {
            Self::LitInt(v) => Some(LosslessConst::Int(v.instantiate())),
            Self::LitSInt(v) => Some(LosslessConst::SInt(v.instantiate())),
            Self::LitUInt(v) => Some(LosslessConst::UInt(v.instantiate())),
            &Self::ImmBool(v) => Some(LosslessConst::Bool(v)),
            &Self::ImmInt(v) => Some(LosslessConst::Int(v)),
            &Self::ImmSInt(v) => Some(LosslessConst::SInt(v)),
            &Self::ImmUInt(v) => Some(LosslessConst::UInt(v)),
            _ => None,
        }
    }

    #[inline]
    pub fn dump(&self, writer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::LitInt(l) => write!(writer, "LitInt {l:?}"),
            Self::LitNum(l) => write!(writer, "LitNum {l:?}"),
            Self::LitSInt(l) => write!(writer, "LitSInt {l:?}"),
            Self::LitUInt(l) => write!(writer, "LitUInt {l:?}"),
            Self::LitFloat(l) => write!(writer, "LitFloat {l:?}"),
            Self::ImmUnit => write!(writer, "()"),
            Self::ImmBool(b) => write!(writer, "{b}"),
            Self::ImmInt(v) => write!(writer, "Int {v}"),
            Self::ImmSInt(v) => write!(writer, "SInt {v}"),
            Self::ImmUInt(v) => write!(writer, "UInt {v}"),
            Self::IntrinsicFunctionRef(overloads) => {
                write!(writer, "ref intrinsicFunction[{overloads:?}]")
            }
            Self::IntrinsicTypeConstructorRef(it) => {
                write!(writer, "ref intrinsicTypeConstructor#{it:?}")
            }
            Self::ScopeLocalVarRef(scope, id) => {
                write!(writer, "ref ScopeLocalVar[#{id} in {scope:?}]")
            }
            Self::FunctionInputVarRef(scope, id) => {
                write!(writer, "ref FunctionInputVar[#{id} in {scope:?}]")
            }
            Self::UserDefinedFunctionRef(scope, name) => {
                write!(writer, "ref UserDefinedFunction[{name:?} in {scope:?}]")
            }
            Self::BuiltinIORef(bio) => write!(writer, "ref BuiltinIO={bio:?}"),
            Self::DescriptorRef { set, binding } => {
                write!(writer, "ref descriptor[set={set}, binding={binding}]")
            }
            Self::PushConstantRef(offset) => write!(writer, "ref pushConstant @ {offset}"),
            Self::WorkgroupSharedMemoryRef(path) => {
                write!(writer, "ref workgroupSharedMemory[codePath={path:?}]")
            }
        }
    }

    #[inline(always)]
    pub const fn typed(self, ty: ConcreteType<'s>) -> TypedBlockConstInstruction<'a, 's> {
        TypedBlockConstInstruction { inst: self, ty }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockPureInstruction<'s> {
    Cast(RegisterRef, ConcreteType<'s>),
    InstantiateIntrinsicTypeClass(RegisterRef, IntrinsicType),
    ConstructIntrinsicComposite(IntrinsicType, Vec<RegisterRef>),
    ConstructTuple(Vec<RegisterRef>),
    ConstructStruct(Vec<RegisterRef>),
    PromoteIntToNumber(RegisterRef),
    IntrinsicBinaryOp(RegisterRef, IntrinsicBinaryOperation, RegisterRef),
    IntrinsicUnaryOp(RegisterRef, IntrinsicUnaryOperation),
    Swizzle(RegisterRef, Vec<usize>),
    SwizzleRef(RegisterRef, Vec<usize>),
    MemberRef(RegisterRef, SourceRefSliceEq<'s>),
    StaticPathRef(RefPath),
    ArrayRef {
        source: RegisterRef,
        index: RegisterRef,
    },
    TupleRef(RegisterRef, usize),
    RegisterAlias(RegisterRef),
    PureIntrinsicCall(&'static str, Vec<RegisterRef>),
    PureFuncall(RegisterRef, Vec<RegisterRef>),
    CompositeInsert {
        value: RegisterRef,
        source: RegisterRef,
        index: usize,
    },
}
impl<'s> BlockPureInstruction<'s> {
    #[inline(always)]
    pub fn apply_register_alias(&mut self, map: &HashMap<RegisterRef, RegisterRef>) -> bool {
        self.relocate_register(|r| {
            while let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    #[inline(always)]
    pub fn apply_parallel_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.relocate_register(|r| {
            if let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    pub fn enumerate_ref_registers(&self, mut reporter: impl FnMut(RegisterRef)) {
        match self {
            Self::StaticPathRef(_) => (),
            &Self::Cast(x, _)
            | &Self::InstantiateIntrinsicTypeClass(x, _)
            | &Self::RegisterAlias(x)
            | &Self::IntrinsicUnaryOp(x, _)
            | &Self::PromoteIntToNumber(x)
            | &Self::Swizzle(x, _)
            | &Self::SwizzleRef(x, _)
            | &Self::MemberRef(x, _)
            | &Self::TupleRef(x, _) => {
                reporter(x);
            }
            &Self::IntrinsicBinaryOp(x, _, y)
            | &Self::ArrayRef {
                source: x,
                index: y,
            } => {
                reporter(x);
                reporter(y);
            }
            Self::ConstructIntrinsicComposite(_, ref xs)
            | Self::ConstructTuple(ref xs)
            | Self::ConstructStruct(ref xs)
            | Self::PureIntrinsicCall(_, ref xs)
            | Self::PureFuncall(_, ref xs) => {
                for x in xs {
                    reporter(*x);
                }
            }
            &Self::CompositeInsert { value, source, .. } => {
                reporter(value);
                reporter(source);
            }
        }
    }

    pub fn relocate_register(&mut self, mut relocator: impl FnMut(&mut RegisterRef)) -> bool {
        match self {
            Self::StaticPathRef(_) => false,
            Self::Cast(ref mut x, _)
            | Self::InstantiateIntrinsicTypeClass(ref mut x, _)
            | Self::RegisterAlias(ref mut x)
            | Self::IntrinsicUnaryOp(ref mut x, _)
            | Self::PromoteIntToNumber(ref mut x)
            | Self::Swizzle(ref mut x, _)
            | Self::SwizzleRef(ref mut x, _)
            | Self::MemberRef(ref mut x, _)
            | Self::TupleRef(ref mut x, _) => {
                let x0 = *x;
                relocator(x);
                *x != x0
            }
            Self::IntrinsicBinaryOp(ref mut x, _, ref mut y)
            | Self::ArrayRef {
                source: ref mut x,
                index: ref mut y,
            } => {
                let (x0, y0) = (*x, *y);
                relocator(x);
                relocator(y);
                *x != x0 || *y != y0
            }
            Self::ConstructIntrinsicComposite(_, ref mut xs)
            | Self::ConstructTuple(ref mut xs)
            | Self::ConstructStruct(ref mut xs)
            | Self::PureIntrinsicCall(_, ref mut xs)
            | Self::PureFuncall(_, ref mut xs) => xs.iter_mut().fold(false, |modified, x| {
                let x0 = *x;
                relocator(x);
                modified || *x != x0
            }),
            Self::CompositeInsert {
                ref mut value,
                ref mut source,
                ..
            } => {
                let (x0, y0) = (*value, *source);
                relocator(value);
                relocator(source);
                x0 != *value || y0 != *source
            }
        }
    }

    pub fn dump(&self, w: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::Cast(src, ty) => write!(w, "{src} as {ty:?}"),
            Self::InstantiateIntrinsicTypeClass(src, ty) => {
                write!(w, "Instantiate {src} as {ty:?}")
            }
            Self::PromoteIntToNumber(value) => {
                write!(w, "PromoteIntToNumber({value})")
            }
            Self::MemberRef(source, name) => write!(w, "ref {source}.({name:?})"),
            Self::StaticPathRef(path) => write!(w, "ref {path:?}"),
            Self::ArrayRef { source, index } => write!(w, "ref {source}[{index}]"),
            Self::TupleRef(source, index) => write!(w, "ref {source}.{index}"),
            Self::SwizzleRef(source, elements) => write!(
                w,
                "ref {source}.{}",
                elements
                    .iter()
                    .map(|x| ['x', 'y', 'z', 'w'][*x])
                    .collect::<String>()
            ),
            Self::Swizzle(source, elements) => write!(
                w,
                "{source}.{}",
                elements
                    .iter()
                    .map(|x| ['x', 'y', 'z', 'w'][*x])
                    .collect::<String>()
            ),
            Self::ConstructIntrinsicComposite(it, args) => write!(
                w,
                "ConstructIntrinsicComposite#{it:?}({})",
                CommaSeparatedWriter(args)
            ),
            Self::ConstructTuple(values) => write!(w, "({})", CommaSeparatedWriter(values)),
            Self::ConstructStruct(values) => write!(w, "{{ {} }}", CommaSeparatedWriter(values)),
            Self::IntrinsicBinaryOp(left, op, right) => {
                write!(w, "{op:?} {left}, {right}")
            }
            Self::IntrinsicUnaryOp(value, op) => write!(w, "{op:?} {value}"),
            Self::RegisterAlias(source) => write!(w, "{source}"),
            Self::PureIntrinsicCall(identifier, args) => write!(
                w,
                "PureIntrinsicCall {identifier}({})",
                CommaSeparatedWriter(args)
            ),
            Self::PureFuncall(source, args) => {
                write!(w, "PureFuncall {source}({})", CommaSeparatedWriter(args))
            }
            Self::CompositeInsert {
                value,
                source,
                index,
            } => write!(
                w,
                "CompositeInsert {source}.{} <- {value}",
                ["x", "y", "z", "w"][*index]
            ),
        }
    }

    #[inline(always)]
    pub const fn typed(self, ty: ConcreteType<'s>) -> TypedBlockPureInstruction<'s> {
        TypedBlockPureInstruction { inst: self, ty }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockInstruction {
    LoadRef(RegisterRef),
    Phi(BTreeMap<BlockRef, RegisterRef>),
}
impl BlockInstruction {
    #[inline(always)]
    pub const fn is_block_dependent(&self) -> bool {
        match self {
            Self::Phi { .. } => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub fn dup_phi_incoming(&mut self, old: BlockRef, new: BlockRef) {
        if let Self::Phi(ref mut incomings) = self {
            if let Some(&r) = incomings.get(&old) {
                incomings.insert(new, r);
            }
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

    #[inline(always)]
    pub fn apply_parallel_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.relocate_register(|r| {
            if let Some(&nr) = map.get(r) {
                *r = nr;
            }
        })
    }

    pub fn enumerate_ref_registers(&self, mut reporter: impl FnMut(RegisterRef)) {
        match self {
            &Self::LoadRef(x) => {
                reporter(x);
            }
            Self::Phi(ref incomings) => {
                for x in incomings.values() {
                    reporter(*x);
                }
            }
        }
    }

    pub fn relocate_register(&mut self, mut relocator: impl FnMut(&mut RegisterRef)) -> bool {
        match self {
            Self::LoadRef(ref mut x) => {
                let x0 = *x;
                relocator(x);
                *x != x0
            }
            Self::Phi(ref mut incomings) => incomings.values_mut().fold(false, |modified, x| {
                let x0 = *x;
                relocator(x);
                modified || *x != x0
            }),
        }
    }

    #[inline]
    pub fn dump(&self, w: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::LoadRef(ptr) => write!(w, "Load {ptr}"),
            Self::Phi(incoming_selectors) => write!(
                w,
                "phi [{}]",
                incoming_selectors
                    .iter()
                    .map(|(BlockRef(from), r)| format!("b{from} -> {r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    #[inline(always)]
    pub const fn typed<'s>(self, ty: ConcreteType<'s>) -> TypedBlockInstruction<'s> {
        TypedBlockInstruction { inst: self, ty }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypedBlockConstInstruction<'a, 's> {
    pub inst: BlockConstInstruction<'a, 's>,
    pub ty: ConcreteType<'s>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypedBlockPureInstruction<'s> {
    pub inst: BlockPureInstruction<'s>,
    pub ty: ConcreteType<'s>,
}
impl<'s> TypedBlockPureInstruction<'s> {
    #[inline(always)]
    pub fn apply_parallel_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.inst.apply_parallel_register_alias(map)
    }

    #[inline(always)]
    pub fn apply_register_alias(&mut self, map: &RegisterAliasMap) -> bool {
        self.inst.apply_register_alias(map)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypedBlockInstruction<'s> {
    pub inst: BlockInstruction,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub eval_impure_registers: HashSet<usize>,
    pub flow: BlockFlowInstruction,
}
impl Block {
    #[inline(always)]
    pub fn empty() -> Self {
        Self {
            eval_impure_registers: HashSet::new(),
            flow: BlockFlowInstruction::Undetermined,
        }
    }

    #[inline(always)]
    pub fn flow_only(flow: BlockFlowInstruction) -> Self {
        Self {
            eval_impure_registers: HashSet::new(),
            flow,
        }
    }

    #[inline(always)]
    pub fn has_block_dependent_instructions(
        &self,
        impure_instructions: &ImpureInstructionMap,
    ) -> bool {
        self.eval_impure_registers
            .iter()
            .any(|r| impure_instructions[r].is_block_dependent())
    }

    #[inline(always)]
    pub fn is_loop_term_block(&self) -> bool {
        matches!(
            self.flow,
            BlockFlowInstruction::Break | BlockFlowInstruction::Continue
        )
    }

    #[inline(always)]
    pub fn is_branch_term_block(&self) -> bool {
        matches!(self.flow, BlockFlowInstruction::ConditionalEnd)
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
    pub fn apply_flow_parallel_register_alias(
        &mut self,
        map: &HashMap<RegisterRef, RegisterRef>,
    ) -> bool {
        self.flow.relocate_register(|r| {
            if let Some(&nr) = map.get(r) {
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
        impure_instructions: &mut ImpureInstructionMap,
        relocator: impl FnMut(&mut RegisterRef) + Copy,
    ) -> bool {
        let mut mod_insts = false;
        for r in self.eval_impure_registers.iter() {
            let modified = impure_instructions
                .get_mut(r)
                .unwrap()
                .relocate_register(relocator);
            mod_insts = mod_insts || modified;
        }
        let mod_flow = self.flow.relocate_register(relocator);

        mod_insts || mod_flow
    }

    /// phiにoldからのものがあったらnewからのものにコピー
    #[inline]
    pub fn dup_phi_incoming(
        &mut self,
        impure_instructions: &mut ImpureInstructionMap,
        old: BlockRef,
        new: BlockRef,
    ) {
        for r in self.eval_impure_registers.iter() {
            impure_instructions
                .get_mut(r)
                .unwrap()
                .dup_phi_incoming(old, new);
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
    pub block_eval_impure_registers: HashSet<usize>,
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
            block_eval_impure_registers: HashSet::new(),
        }
    }

    #[inline(always)]
    pub fn into_block(self, flow: BlockFlowInstruction) -> Block {
        Block {
            eval_impure_registers: self.block_eval_impure_registers,
            flow,
        }
    }

    #[inline(always)]
    pub fn create_block(self, flow: BlockFlowInstruction) -> BlockRef {
        let blk = Block {
            eval_impure_registers: self.block_eval_impure_registers,
            flow,
        };

        self.generation_context.add(blk)
    }

    pub fn into_eval_impure_registers(self) -> HashSet<usize> {
        self.block_eval_impure_registers
    }

    #[inline]
    pub fn loaded(&mut self, ptr: RegisterRef) -> RegisterRef {
        match &*ptr.ty(self) {
            ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => self.load_ref(ptr),
            _ => ptr,
        }
    }

    pub fn cast(&mut self, src: RegisterRef, to: ConcreteType<'s>) -> RegisterRef {
        RegisterRef::Pure(
            self.instruction_emission_context
                .add_pure_instruction(BlockPureInstruction::Cast(src, to.clone()).typed(to)),
        )
    }

    pub fn instantiate_intrinsic_type_class(
        &mut self,
        src: RegisterRef,
        to: IntrinsicType,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::InstantiateIntrinsicTypeClass(src, to).typed(to.into()),
        ))
    }

    pub fn load_ref(&mut self, ptr: RegisterRef) -> RegisterRef {
        let dest_register = self.instruction_emission_context.add_impure_instruction(
            BlockInstruction::LoadRef(ptr).typed(ptr.ty(self).as_dereferenced().unwrap().clone()),
        );
        self.block_eval_impure_registers.insert(dest_register);

        RegisterRef::Impure(dest_register)
    }

    pub fn construct_intrinsic_composite(
        &mut self,
        it: IntrinsicType,
        args: Vec<RegisterRef>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::ConstructIntrinsicComposite(it, args).typed(it.into()),
        ))
    }

    pub fn construct_tuple(&mut self, elements: Vec<RegisterRef>) -> RegisterRef {
        let ty = ConcreteType::Tuple(elements.iter().map(|r| r.ty(self).clone()).collect());

        RegisterRef::Pure(
            self.instruction_emission_context
                .add_pure_instruction(BlockPureInstruction::ConstructTuple(elements).typed(ty)),
        )
    }

    pub fn construct_struct(
        &mut self,
        elements: Vec<RegisterRef>,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(
            self.instruction_emission_context.add_pure_instruction(
                BlockPureInstruction::ConstructStruct(elements).typed(out_type),
            ),
        )
    }

    pub fn promote_int_to_number(&mut self, r: RegisterRef) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::PromoteIntToNumber(r).typed(ConcreteType::UnknownNumberClass),
        ))
    }

    pub fn intrinsic_binary_op(
        &mut self,
        left: RegisterRef,
        op: IntrinsicBinaryOperation,
        right: RegisterRef,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::IntrinsicBinaryOp(left, op, right).typed(out_type),
        ))
    }

    pub fn intrinsic_unary_op(
        &mut self,
        value: RegisterRef,
        op: IntrinsicUnaryOperation,
        out_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::IntrinsicUnaryOp(value, op).typed(out_type),
        ))
    }

    pub fn intrinsic_function_ref(
        &mut self,
        overloads: Vec<IntrinsicFunctionSymbol>,
    ) -> RegisterRef {
        let ty = ConcreteType::OverloadedFunctions(
            overloads
                .iter()
                .map(|x| (x.args.clone(), Box::new(x.output.clone())))
                .collect(),
        );

        RegisterRef::Const(
            self.instruction_emission_context
                .add_constant(BlockConstInstruction::IntrinsicFunctionRef(overloads).typed(ty)),
        )
    }

    pub fn intrinsic_type_constructor_ref(&mut self, it: IntrinsicType) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::IntrinsicTypeConstructorRef(it)
                    .typed(ConcreteType::IntrinsicTypeConstructor(it)),
            ),
        )
    }

    pub fn scope_local_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().imm_ref();

        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::ScopeLocalVarRef(PtrEq(scope), var_id).typed(ty),
            ),
        )
    }

    pub fn scope_local_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.local_vars.borrow()[var_id].ty.clone().mutable_ref();

        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::ScopeLocalVarRef(PtrEq(scope), var_id).typed(ty),
            ),
        )
    }

    pub fn function_input_var_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().imm_ref();

        RegisterRef::Const(self.instruction_emission_context.add_constant(
            BlockConstInstruction::FunctionInputVarRef(PtrEq(scope), var_id).typed(ty),
        ))
    }

    pub fn function_input_var_mutable_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        var_id: usize,
    ) -> RegisterRef {
        let ty = scope.function_input_vars[var_id].ty.clone().mutable_ref();

        RegisterRef::Const(self.instruction_emission_context.add_constant(
            BlockConstInstruction::FunctionInputVarRef(PtrEq(scope), var_id).typed(ty),
        ))
    }

    pub fn user_defined_function_ref(
        &mut self,
        scope: &'a SymbolScope<'a, 's>,
        name: SourceRef<'s>,
    ) -> RegisterRef {
        let fs = scope.user_defined_function_symbol(name.slice).unwrap();
        let ty = ConcreteType::Function {
            args: fs.inputs.iter().map(|(_, _, _, t)| t.clone()).collect(),
            output: match &fs.output[..] {
                &[] => None,
                &[(_, ref t)] => Some(Box::new(t.clone())),
                xs => Some(Box::new(ConcreteType::Tuple(
                    xs.iter().map(|(_, t)| t.clone()).collect(),
                ))),
            },
        };

        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::UserDefinedFunctionRef(PtrEq(scope), SourceRefSliceEq(name))
                    .typed(ty),
            ),
        )
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

        RegisterRef::Pure(
            self.instruction_emission_context
                .add_pure_instruction(BlockPureInstruction::Swizzle(source, elements).typed(ty)),
        )
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

        RegisterRef::Pure(
            self.instruction_emission_context
                .add_pure_instruction(BlockPureInstruction::SwizzleRef(source, elements).typed(ty)),
        )
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

        RegisterRef::Pure(
            self.instruction_emission_context
                .add_pure_instruction(BlockPureInstruction::SwizzleRef(source, elements).typed(ty)),
        )
    }

    pub fn member_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(
            self.instruction_emission_context.add_pure_instruction(
                BlockPureInstruction::MemberRef(source, SourceRefSliceEq(name))
                    .typed(member_type.imm_ref()),
            ),
        )
    }

    pub fn member_mutable_ref(
        &mut self,
        source: RegisterRef,
        name: SourceRef<'s>,
        member_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(
            self.instruction_emission_context.add_pure_instruction(
                BlockPureInstruction::MemberRef(source, SourceRefSliceEq(name))
                    .typed(member_type.mutable_ref()),
            ),
        )
    }

    pub fn array_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::ArrayRef { source, index }.typed(element_type.imm_ref()),
        ))
    }

    pub fn array_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: RegisterRef,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::ArrayRef { source, index }.typed(element_type.mutable_ref()),
        ))
    }

    pub fn tuple_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::TupleRef(source, index).typed(element_type.imm_ref()),
        ))
    }

    pub fn tuple_mutable_ref(
        &mut self,
        source: RegisterRef,
        index: usize,
        element_type: ConcreteType<'s>,
    ) -> RegisterRef {
        RegisterRef::Pure(self.instruction_emission_context.add_pure_instruction(
            BlockPureInstruction::TupleRef(source, index).typed(element_type.mutable_ref()),
        ))
    }

    pub fn const_int(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::LitInt(ConstIntLiteral(
                    SourceRefSliceEq(repr),
                    ConstModifiers::empty(),
                ))
                .typed(ConcreteType::UnknownIntClass),
            ),
        )
    }

    pub fn const_number(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::LitNum(ConstNumberLiteral(
                    SourceRefSliceEq(repr),
                    ConstModifiers::empty(),
                ))
                .typed(ConcreteType::UnknownNumberClass),
            ),
        )
    }

    pub fn const_uint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::LitUInt(ConstUIntLiteral(
                    SourceRefSliceEq(repr),
                    ConstModifiers::empty(),
                ))
                .typed(IntrinsicType::UInt.into()),
            ),
        )
    }

    pub fn const_sint(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::LitSInt(ConstSIntLiteral(
                    SourceRefSliceEq(repr),
                    ConstModifiers::empty(),
                ))
                .typed(IntrinsicType::SInt.into()),
            ),
        )
    }

    pub fn const_float(&mut self, repr: SourceRef<'s>) -> RegisterRef {
        RegisterRef::Const(
            self.instruction_emission_context.add_constant(
                BlockConstInstruction::LitFloat(ConstFloatLiteral(
                    SourceRefSliceEq(repr),
                    ConstModifiers::empty(),
                ))
                .typed(IntrinsicType::Float.into()),
            ),
        )
    }

    pub fn phi(
        &mut self,
        incoming_selectors: BTreeMap<BlockRef, RegisterRef>,
        ty: ConcreteType<'s>,
    ) -> RegisterRef {
        let dest_register = self
            .instruction_emission_context
            .add_impure_instruction(BlockInstruction::Phi(incoming_selectors).typed(ty));
        self.block_eval_impure_registers.insert(dest_register);

        RegisterRef::Impure(dest_register)
    }
}

pub struct BlockInstructionEmissionContext<'a, 's> {
    pub constants: Constants<'a, 's>,
    pub pure_instructions: PureInstructions<'s>,
    pub impure_registers: Vec<ConcreteType<'s>>,
    pub impure_instructions: ImpureInstructionMap,
}
impl<'c, 'a, 's> RegisterTypeProvider<'c, 's> for BlockInstructionEmissionContext<'a, 's> {
    #[inline(always)]
    fn register_type(&'c self, register: RegisterRef) -> &'c ConcreteType<'s> {
        match register {
            RegisterRef::Const(n) => &self.constants[n].ty,
            RegisterRef::Pure(n) => &self.pure_instructions[n].ty,
            RegisterRef::Impure(n) => &self.impure_registers[n],
        }
    }
}
impl<'a, 's> BlockInstructionEmissionContext<'a, 's> {
    pub fn new() -> Self {
        Self {
            constants: Vec::new(),
            impure_registers: Vec::new(),
            pure_instructions: Vec::new(),
            impure_instructions: HashMap::new(),
        }
    }

    #[inline]
    pub fn add_constant(&mut self, c: TypedBlockConstInstruction<'a, 's>) -> usize {
        self.constants.push(c);
        self.constants.len() - 1
    }

    #[inline]
    fn add_pure_instruction(&mut self, inst: TypedBlockPureInstruction<'s>) -> usize {
        self.pure_instructions.push(inst);
        self.pure_instructions.len() - 1
    }

    #[inline]
    pub fn alloc_impure_register(&mut self, ty: ConcreteType<'s>) -> usize {
        self.impure_registers.push(ty);
        self.impure_registers.len() - 1
    }

    #[inline]
    fn add_impure_instruction(&mut self, inst: TypedBlockInstruction<'s>) -> usize {
        self.impure_registers.push(inst.ty);
        self.impure_instructions
            .insert(self.impure_registers.len() - 1, inst.inst);
        self.impure_registers.len() - 1
    }

    #[inline]
    pub fn const_unit(&mut self) -> RegisterRef {
        RegisterRef::Const(
            self.add_constant(BlockConstInstruction::ImmUnit.typed(IntrinsicType::Unit.into())),
        )
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
        constants: &Constants<'a, 's>,
        pure_instructions: &PureInstructions<'s>,
        impure_registers: &[ConcreteType<'s>],
        impure_instructions: &ImpureInstructionMap,
    ) -> std::io::Result<()> {
        writeln!(writer, "Registers: ")?;
        dump_registers(writer, constants, pure_instructions, impure_registers)?;
        dump_blocks(writer, &self.blocks, impure_instructions)?;

        Ok(())
    }
}

#[repr(transparent)]
struct CommaSeparatedWriter<'a, T: 'a>(pub &'a [T]);
impl<'a, T: 'a + core::fmt::Display> core::fmt::Display for CommaSeparatedWriter<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut written = false;
        for e in self.0.iter() {
            if written {
                write!(f, ", {e}")?;
            } else {
                e.fmt(f)?;
            }

            written = true;
        }

        Ok(())
    }
}

pub fn dump_registers(
    writer: &mut (impl std::io::Write + ?Sized),
    constants: &Constants,
    pure_instructions: &PureInstructions,
    impure_registers: &[ConcreteType],
) -> std::io::Result<()> {
    for (n, x) in constants.iter().enumerate() {
        write!(writer, "  c{n}: {:?} = ", x.ty)?;
        x.inst.dump(writer)?;
        writeln!(writer)?;
    }
    for (n, x) in pure_instructions.iter().enumerate() {
        write!(writer, "  x{n}: {:?} = ", x.ty)?;
        x.inst.dump(writer)?;
        writeln!(writer)?;
    }
    for (n, r) in impure_registers.iter().enumerate() {
        writeln!(writer, "  r{n}: {r:?}")?;
    }

    Ok(())
}

pub fn dump_blocks(
    writer: &mut (impl std::io::Write + ?Sized),
    blocks: &[Block],
    impure_instructions: &ImpureInstructionMap,
) -> std::io::Result<()> {
    for (n, b) in blocks.iter().enumerate() {
        writeln!(writer, "b{n}: {{")?;

        let mut sorted_evals = b.eval_impure_registers.iter().collect::<Vec<_>>();
        sorted_evals.sort();
        if !sorted_evals.is_empty() {
            for r in sorted_evals {
                write!(writer, "  r{r} = ")?;
                impure_instructions[r].dump(writer)?;
                writer.write(b"\n")?;
            }

            writer.write(b"\n")?;
        }

        match b.flow {
            BlockFlowInstruction::Goto(BlockRef(x)) => writeln!(writer, "  goto -> b{x}")?,
            BlockFlowInstruction::StoreRef {
                ptr,
                value,
                after: Some(BlockRef(after)),
            } => writeln!(writer, "  *{ptr} = {value} -> b{after}")?,
            BlockFlowInstruction::StoreRef {
                ptr,
                value,
                after: None,
            } => writeln!(writer, "  *{ptr} = {value} -> ???")?,
            BlockFlowInstruction::Funcall {
                callee,
                ref args,
                result,
                after_return: Some(BlockRef(after_return)),
            } => writeln!(
                writer,
                "  {result} = ({callee})({}) -> b{after_return}",
                CommaSeparatedWriter(args)
            )?,
            BlockFlowInstruction::Funcall {
                callee,
                ref args,
                result,
                after_return: None,
            } => writeln!(
                writer,
                "  {result} = ({callee})({}) -> ???",
                CommaSeparatedWriter(args)
            )?,
            BlockFlowInstruction::IntrinsicImpureFuncall {
                identifier,
                ref args,
                result,
                after_return: Some(BlockRef(after_return)),
            } => writeln!(
                writer,
                "  {result} = IntrinsicImpureFuncall {identifier}({}) -> b{after_return}",
                CommaSeparatedWriter(args)
            )?,
            BlockFlowInstruction::IntrinsicImpureFuncall {
                identifier,
                ref args,
                result,
                after_return: None,
            } => writeln!(
                writer,
                "  {result} = IntrinsicImpureFuncall {identifier}({}) -> ???",
                CommaSeparatedWriter(args)
            )?,
            BlockFlowInstruction::Conditional {
                source,
                r#true: BlockRef(t),
                r#false: BlockRef(e),
                merge: BlockRef(m),
            } => writeln!(
                writer,
                "  branch {source} ? -> b{t} : -> b{e} merged at b{m}"
            )?,
            BlockFlowInstruction::ConditionalEnd => {
                writeln!(writer, "  -- conditional branch end")?
            }
            BlockFlowInstruction::Return(r) => writeln!(writer, "  return {r}")?,
            BlockFlowInstruction::ConditionalLoop {
                condition,
                r#break: BlockRef(brk),
                r#continue: BlockRef(cont),
                body: BlockRef(body),
            } => writeln!(
                writer,
                "  loop while {condition} ... break -> b{brk}, continue -> b{cont}, body -> b{body}"
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
                        match &*left_ref_expr.result.ty(inst.instruction_emission_context) {
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
            let right_value = match &*left_ref_expr.result.ty(inst.instruction_emission_context) {
                ConcreteType::MutableRef(inner) => {
                    match (
                        &*right_value.ty(inst.instruction_emission_context),
                        &**inner,
                    ) {
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

#[derive(Debug, Clone)]
pub struct BlockifiedProgram<'a, 's> {
    pub blocks: Vec<Block>,
    pub impure_registers: Vec<ConcreteType<'s>>,
    pub impure_instructions: ImpureInstructionMap,
    pub pure_instructions: PureInstructions<'s>,
    pub constants: Constants<'a, 's>,
}
impl<'a, 's> BlockifiedProgram<'a, 's> {
    pub fn dump(&self, writer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        writeln!(writer, "Registers: ")?;
        dump_registers(
            writer,
            &self.constants,
            &self.pure_instructions,
            &self.impure_registers,
        )?;
        dump_blocks(writer, &self.blocks, &self.impure_instructions)?;

        Ok(())
    }

    /// 一連のブロックを接続しつつ末尾に追加する
    /// 返り値: 追加されたBlockRefの範囲
    pub fn append_block_sequence(
        &mut self,
        seq: impl IntoIterator<Item = Block>,
    ) -> RangeInclusive<BlockRef> {
        let first_block_ref = BlockRef(self.blocks.len());
        let blocks_iter = seq.into_iter();
        let block_size_hint = blocks_iter.size_hint();
        self.blocks
            .reserve(block_size_hint.1.unwrap_or(block_size_hint.0));
        let mut last_block_ref: Option<BlockRef> = None;
        for (oref, b) in blocks_iter.enumerate() {
            let new_block_ref = BlockRef(first_block_ref.0 + oref);
            if let Some(lb) = last_block_ref {
                assert!(self.blocks[lb.0].try_set_next(new_block_ref));
            }
            self.blocks.push(b);
            last_block_ref = Some(new_block_ref);
        }

        first_block_ref..=last_block_ref.unwrap_or(first_block_ref)
    }

    #[inline]
    pub fn add_evaluated_impure_instruction(
        &mut self,
        impure_instruction: TypedBlockInstruction<'s>,
        eval_at: BlockRef,
    ) -> RegisterRef {
        let r = self.add_impure_instruction(impure_instruction);
        self.blocks[eval_at.0]
            .eval_impure_registers
            .insert(r.as_id());

        r
    }

    pub fn add_impure_instruction(
        &mut self,
        impure_instruction: TypedBlockInstruction<'s>,
    ) -> RegisterRef {
        self.impure_registers.push(impure_instruction.ty);
        self.impure_instructions
            .insert(self.impure_registers.len() - 1, impure_instruction.inst);

        RegisterRef::Impure(self.impure_registers.len() - 1)
    }

    pub fn add_pure_instruction(
        &mut self,
        pure_instruction: TypedBlockPureInstruction<'s>,
    ) -> RegisterRef {
        self.pure_instructions.push(pure_instruction);

        RegisterRef::Pure(self.pure_instructions.len() - 1)
    }

    pub fn add_constant(&mut self, c: TypedBlockConstInstruction<'a, 's>) -> RegisterRef {
        self.constants.push(c);

        RegisterRef::Const(self.constants.len() - 1)
    }

    pub fn apply_parallel_register_alias(&mut self, alias_map: &RegisterAliasMap) {
        if alias_map.is_empty() {
            // レジスタエイリアスなし
            return;
        }

        println!("[Register Alias(Parallel)]");
        let mut sorted_alias = alias_map.iter().collect::<Vec<_>>();
        sorted_alias.sort_by(|(a, _), (b, _)| a.entropy_order(b));
        for (from, to) in sorted_alias {
            println!("  {from:?} -> {to:?}");
        }

        for x in self.impure_instructions.values_mut() {
            x.apply_parallel_register_alias(alias_map);
        }
        for x in self.pure_instructions.iter_mut() {
            x.apply_parallel_register_alias(alias_map);
        }
        for b in self.blocks.iter_mut() {
            b.apply_flow_parallel_register_alias(alias_map);
        }
    }

    pub fn apply_register_alias(&mut self, alias_map: &RegisterAliasMap) {
        if alias_map.is_empty() {
            // レジスタエイリアスなし
            return;
        }

        println!("[Register Alias(ResolveChained)]");
        let mut sorted_alias = alias_map.iter().collect::<Vec<_>>();
        sorted_alias.sort_by(|(a, _), (b, _)| a.entropy_order(b));
        for (from, to) in sorted_alias {
            println!("  {from:?} -> {to:?}");
        }

        for x in self.impure_instructions.values_mut() {
            x.apply_register_alias(alias_map);
        }
        for x in self.pure_instructions.iter_mut() {
            x.apply_register_alias(alias_map);
        }
        for b in self.blocks.iter_mut() {
            b.apply_flow_register_alias(alias_map);
        }
    }

    /// どこからも参照されていないPureRegisterを収集する
    pub fn collect_unreferenced_pure_registers(&self) -> HashSet<usize> {
        let mut unreferenced = (0..self.pure_instructions.len()).collect::<HashSet<_>>();

        for x in self.pure_instructions.iter() {
            x.inst.enumerate_ref_registers(|r| {
                if let RegisterRef::Pure(n) = r {
                    unreferenced.remove(&n);
                }
            });
        }
        for x in self.impure_instructions.values() {
            x.enumerate_ref_registers(|r| {
                if let RegisterRef::Pure(n) = r {
                    unreferenced.remove(&n);
                }
            });
        }
        for b in self.blocks.iter() {
            b.flow.enumerate_ref_registers(|r| {
                if let RegisterRef::Pure(n) = r {
                    unreferenced.remove(&n);
                }
            })
        }

        unreferenced
    }

    /// どこからも参照されていないImpureRegisterを収集する
    pub fn collect_unreferenced_impure_registers(&self) -> HashSet<usize> {
        let mut unreferenced = (0..self.impure_registers.len()).collect::<HashSet<_>>();

        for x in self.pure_instructions.iter() {
            x.inst.enumerate_ref_registers(|r| {
                if let RegisterRef::Impure(n) = r {
                    unreferenced.remove(&n);
                }
            });
        }
        for x in self.impure_instructions.values() {
            x.enumerate_ref_registers(|r| {
                if let RegisterRef::Impure(n) = r {
                    unreferenced.remove(&n);
                }
            });
        }
        for b in self.blocks.iter() {
            b.flow.enumerate_ref_registers(|r| {
                if let RegisterRef::Impure(n) = r {
                    unreferenced.remove(&n);
                }
            });
            // Note: Flowのdestはかならずあるものとする（Impureな関数呼び出しはstripしてはいけない）
            b.flow.enumerate_dest_register(|r| {
                if let RegisterRef::Impure(n) = r {
                    unreferenced.remove(&n);
                }
            })
        }

        unreferenced
    }

    /// Pureなレジスタエイリアス命令を抽出する
    pub fn collect_pure_register_aliases(&self) -> RegisterAliasMap {
        self.pure_instructions
            .iter()
            .enumerate()
            .filter_map(|(from, x)| match x.inst {
                BlockPureInstruction::RegisterAlias(to) => Some((RegisterRef::Pure(from), to)),
                _ => None,
            })
            .collect()
    }
}
impl<'c, 'a, 's> RegisterTypeProvider<'c, 's> for BlockifiedProgram<'a, 's> {
    fn register_type(&'c self, register: RegisterRef) -> &'c ConcreteType<'s> {
        match register {
            RegisterRef::Const(x) => &self.constants[x].ty,
            RegisterRef::Pure(x) => &self.pure_instructions[x].ty,
            RegisterRef::Impure(x) => &self.impure_registers[x],
        }
    }
}

pub fn parse_incoming_flows(blocks: &[Block]) -> HashMap<BlockRef, Vec<BlockRef>> {
    let mut incomings = HashMap::new();
    let mut processed = HashSet::new();
    let mut loop_stack = Vec::new();
    let mut merge_stack = Vec::new();

    fn parse(
        blocks: &[Block],
        n: BlockRef,
        incomings: &mut HashMap<BlockRef, Vec<BlockRef>>,
        processed: &mut HashSet<BlockRef>,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
        merge_stack: &mut Vec<BlockRef>,
    ) {
        if processed.contains(&n) {
            return;
        }

        processed.insert(n);
        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                incomings.entry(next).or_insert_with(Vec::new).push(n);
                parse(blocks, next, incomings, processed, loop_stack, merge_stack);
            }
            BlockFlowInstruction::StoreRef {
                after: Some(after), ..
            } => {
                incomings.entry(after).or_insert_with(Vec::new).push(n);
                parse(blocks, after, incomings, processed, loop_stack, merge_stack);
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
                parse(
                    blocks,
                    after_return,
                    incomings,
                    processed,
                    loop_stack,
                    merge_stack,
                );
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
                parse(
                    blocks,
                    after_return,
                    incomings,
                    processed,
                    loop_stack,
                    merge_stack,
                );
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
            BlockFlowInstruction::Conditional {
                r#true,
                r#false,
                merge,
                ..
            } => {
                incomings.entry(r#true).or_insert_with(Vec::new).push(n);
                incomings.entry(r#false).or_insert_with(Vec::new).push(n);
                merge_stack.push(merge);
                parse(
                    blocks,
                    r#true,
                    incomings,
                    processed,
                    loop_stack,
                    &mut merge_stack.clone(),
                );
                parse(
                    blocks,
                    r#false,
                    incomings,
                    processed,
                    loop_stack,
                    &mut merge_stack.clone(),
                );
                merge_stack.pop();
            }
            BlockFlowInstruction::ConditionalEnd => {
                let m = merge_stack.pop().expect("not in conditional branch");
                incomings.entry(m).or_insert_with(Vec::new).push(n);
                parse(blocks, m, incomings, processed, loop_stack, merge_stack);
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
                parse(blocks, body, incomings, processed, loop_stack, merge_stack);
                parse(
                    blocks,
                    r#break,
                    incomings,
                    processed,
                    loop_stack,
                    merge_stack,
                );
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
        &mut merge_stack,
    );
    incomings
}
