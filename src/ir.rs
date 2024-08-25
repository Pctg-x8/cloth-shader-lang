use std::collections::HashMap;

use block::{Block, BlockInstruction, RegisterRef};
use expr::ConstModifiers;

use crate::{concrete_type::ConcreteType, scope::SymbolScope, source_ref::SourceRefSliceEq};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstantiatedConst {
    Bool(bool),
    Int(isize),
    SInt(i32),
    UInt(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstIntLiteral<'s>(pub SourceRefSliceEq<'s>, pub ConstModifiers);
impl<'s> ConstIntLiteral<'s> {
    pub fn instantiate(&self) -> isize {
        let base = if let Some(rest) = self
            .0
             .0
            .slice
            .strip_prefix("0x")
            .or_else(|| self.0 .0.slice.strip_prefix("0X"))
        {
            // hex
            isize::from_str_radix(rest, 16).unwrap()
        } else {
            self.0 .0.slice.parse::<isize>().unwrap()
        };

        let base = if self.1.contains(ConstModifiers::NEGATE) {
            -base
        } else {
            base
        };
        let base = if self.1.contains(ConstModifiers::BIT_NOT) {
            !base
        } else {
            base
        };
        // TODO: logical_not(これはboolになるのでConstModifiersでやらないほうがいいかも)

        base
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstSIntLiteral<'s>(pub SourceRefSliceEq<'s>, pub ConstModifiers);
impl<'s> ConstSIntLiteral<'s> {
    pub fn instantiate(&self) -> i32 {
        let base = if let Some(rest) = self
            .0
             .0
            .slice
            .strip_prefix("0x")
            .or_else(|| self.0 .0.slice.strip_prefix("0X"))
        {
            // hex
            i32::from_str_radix(rest, 16).unwrap()
        } else {
            self.0 .0.slice.parse::<i32>().unwrap()
        };

        let base = if self.1.contains(ConstModifiers::NEGATE) {
            -base
        } else {
            base
        };
        let base = if self.1.contains(ConstModifiers::BIT_NOT) {
            !base
        } else {
            base
        };
        // TODO: logical_not(これはboolになるのでConstModifiersでやらないほうがいいかも)

        base
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstUIntLiteral<'s>(pub SourceRefSliceEq<'s>, pub ConstModifiers);
impl<'s> ConstUIntLiteral<'s> {
    pub fn instantiate(&self) -> u32 {
        let base = if let Some(rest) = self
            .0
             .0
            .slice
            .strip_prefix("0x")
            .or_else(|| self.0 .0.slice.strip_prefix("0X"))
        {
            // hex
            u32::from_str_radix(rest, 16).unwrap()
        } else {
            self.0 .0.slice.parse::<u32>().unwrap()
        };

        let base = if self.1.contains(ConstModifiers::NEGATE) {
            panic!("Error: cannot apply negate for unsigned value")
        } else {
            base
        };
        let base = if self.1.contains(ConstModifiers::BIT_NOT) {
            !base
        } else {
            base
        };
        // TODO: logical_not(これはboolになるのでConstModifiersでやらないほうがいいかも)

        base
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstNumberLiteral<'s>(pub SourceRefSliceEq<'s>, pub ConstModifiers);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstFloatLiteral<'s>(pub SourceRefSliceEq<'s>, pub ConstModifiers);
impl<'s> ConstFloatLiteral<'s> {
    pub fn instantiate(&self) -> f32 {
        let base = if let Some(rest) = self.0 .0.slice.strip_suffix(['f', 'F']) {
            rest.parse::<f32>().unwrap()
        } else {
            self.0 .0.slice.parse::<f32>().unwrap()
        };

        let base = if self.1.contains(ConstModifiers::NEGATE) {
            -base
        } else {
            base
        };
        let base = if self.1.contains(ConstModifiers::BIT_NOT) {
            panic!("Error: cannot apply bit not for float literal")
        } else {
            base
        };
        // TODO: logical_not(これはboolになるのでConstModifiersでやらないほうがいいかも)

        base
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBody<'a, 's> {
    pub symbol_scope: &'a SymbolScope<'a, 's>,
    pub registers: Vec<ConcreteType<'s>>,
    pub constants: HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    pub blocks: Vec<Block<'a, 's>>,
}
