use expr::SimplifiedExpression;

use crate::{concrete_type::ConcreteType, scope::SymbolScope};

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
    pub expressions: Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    pub returning: ExprRef,
    pub returning_type: ConcreteType<'s>,
    pub is_referential_transparent: bool,
}
