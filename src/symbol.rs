use meta::SymbolAttribute;

use crate::{concrete_type::ConcreteType, ir::ExprRef, source_ref::SourceRef};

pub mod meta;

#[derive(Debug, Clone)]
pub struct IntrinsicFunctionSymbol {
    pub name: &'static str,
    pub ty: ConcreteType<'static>,
    pub is_pure: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionInputVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub struct LocalVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub init_expr_id: ExprRef,
}

#[derive(Debug, Clone)]
pub struct UserDefinedFunctionSymbol<'s> {
    pub occurence: SourceRef<'s>,
    pub attribute: SymbolAttribute,
    pub inputs: Vec<(SymbolAttribute, SourceRef<'s>, ConcreteType<'s>)>,
    pub output: Vec<(SymbolAttribute, ConcreteType<'s>)>,
}
