use block::Block;
use expr::{SimplifiedExpression, TypedExprRef};

use crate::{concrete_type::ConcreteType, scope::SymbolScope};

pub mod block;
pub mod expr;
pub mod opt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ExprRef(pub usize);
impl core::fmt::Debug for ExprRef {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBody<'a, 's> {
    pub symbol_scope: &'a SymbolScope<'a, 's>,
    pub registers: Vec<ConcreteType<'s>>,
    pub blocks: Vec<Block<'a, 's>>,
}
