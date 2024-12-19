use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::parser::ExpressionNode;

pub fn reduce_const_expr(expr: &ExpressionNode) -> Constant {
    match expr {
        ExpressionNode::Number(x) => {
            let is_hex = x.slice.starts_with("0x") || x.slice.starts_with("0X");
            let is_floating = x.slice.ends_with(['f', 'F']);
            let is_number = x.slice.contains('.');

            if is_hex {
                Constant::UnknownInt(x.slice.parse().unwrap())
            } else if is_floating {
                Constant::Float(x.slice.parse().unwrap())
            } else if is_number {
                Constant::UnknownNumber(x.slice.parse().unwrap())
            } else {
                Constant::UnknownInt(x.slice.parse().unwrap())
            }
        }
        ExpressionNode::Binary(l, op, r) => match op.slice {
            "+" => reduce_const_expr(l) + reduce_const_expr(r),
            "-" => reduce_const_expr(l) - reduce_const_expr(r),
            "*" => reduce_const_expr(l) * reduce_const_expr(r),
            "/" => reduce_const_expr(l) / reduce_const_expr(r),
            "%" => reduce_const_expr(l) % reduce_const_expr(r),
            "&" => reduce_const_expr(l)
                .try_bit_and(reduce_const_expr(r))
                .expect("cannot perform bitand op"),
            "|" => reduce_const_expr(l)
                .try_bit_or(reduce_const_expr(r))
                .expect("cannot perform bitor op"),
            "^" => reduce_const_expr(l)
                .try_bit_xor(reduce_const_expr(r))
                .expect("cannot perform bitxor op"),
            "==" => Constant::Bool(reduce_const_expr(l) == reduce_const_expr(r)),
            "!=" => Constant::Bool(reduce_const_expr(l) != reduce_const_expr(r)),
            "<" => Constant::Bool(reduce_const_expr(l) < reduce_const_expr(r)),
            ">" => Constant::Bool(reduce_const_expr(l) > reduce_const_expr(r)),
            "<=" => Constant::Bool(reduce_const_expr(l) <= reduce_const_expr(r)),
            ">=" => Constant::Bool(reduce_const_expr(l) >= reduce_const_expr(r)),
            "&&" => Constant::Bool(reduce_const_expr(l).logical_and(reduce_const_expr(r))),
            "||" => Constant::Bool(reduce_const_expr(l).logical_or(reduce_const_expr(r))),
            "^^" => reduce_const_expr(l)
                .try_pow(reduce_const_expr(r))
                .expect("cannot perform pow op"),
            _ => unimplemented!("unknown op {op:?}"),
        },
        _ => panic!("Error: cannot reduce expression to constant"),
    }
}

/// 定数
#[derive(Debug, Clone, Copy)]
pub enum Constant {
    /// ブール
    Bool(bool),
    /// 符号なし32bit整数
    UInt(u32),
    /// 符号付き32bit整数
    SInt(i32),
    /// 32bit浮動小数点数
    Float(f32),
    /// 型明示なし整数
    UnknownInt(isize),
    /// 型明示なし実数
    UnknownNumber(f64),
}
impl Constant {
    /// boolにする
    pub const fn into_bool(self) -> bool {
        match self {
            Self::Bool(x) => x,
            Self::UInt(x) => x != 0,
            Self::SInt(x) => x != 0,
            Self::Float(x) => x != 0.0,
            Self::UnknownInt(x) => x != 0,
            Self::UnknownNumber(x) => x != 0.0,
        }
    }

    /// 符号なし32bit整数にする
    pub const fn into_u32(self) -> u32 {
        match self {
            Self::Bool(true) => 1,
            Self::Bool(false) => 0,
            Self::UInt(x) => x,
            Self::SInt(x) => x as _,
            Self::Float(x) => x as _,
            Self::UnknownInt(x) => x as _,
            Self::UnknownNumber(x) => x as _,
        }
    }
}
impl Add for Constant {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Self::UInt(if l { 1 } else { 0 } + if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(if l { 1 } else { 0 } + r),
                Self::SInt(r) => Self::SInt(if l { 1 } else { 0 } + r),
                Self::Float(r) => Self::Float(if l { 1.0 } else { 0.0 } + r),
                Self::UnknownInt(r) => Self::UnknownInt(if l { 1 } else { 0 } + r),
                Self::UnknownNumber(r) => Self::UnknownNumber(if l { 1.0 } else { 0.0 } + r),
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Self::UInt(l + if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(l + r),
                Self::SInt(r) => Self::SInt(l as i32 + r),
                Self::Float(r) => Self::Float(l as f32 + r),
                Self::UnknownInt(r) => Self::UInt(l + r as u32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 + r),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l + if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l + r as i32),
                Self::SInt(r) => Self::SInt(l + r),
                Self::Float(r) => Self::Float(l as f32 + r),
                Self::UnknownInt(r) => Self::SInt(l + r as i32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 + r),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Self::Float(l + if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l + r as f32),
                Self::SInt(r) => Self::Float(l + r as f32),
                Self::Float(r) => Self::Float(l + r),
                Self::UnknownInt(r) => Self::Float(l + r as f32),
                Self::UnknownNumber(r) => Self::Float(l + r as f32),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l as i32 + if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l as i32 + r as i32),
                Self::SInt(r) => Self::SInt(l as i32 + r),
                Self::Float(r) => Self::Float(l as f32 + r),
                Self::UnknownInt(r) => Self::UnknownInt(l + r),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 + r),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Self::Float(l as f32 + if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l as f32 + r as f32),
                Self::SInt(r) => Self::Float(l as f32 + r as f32),
                Self::Float(r) => Self::Float(l as f32 + r),
                Self::UnknownInt(r) => Self::UnknownNumber(l + r as f64),
                Self::UnknownNumber(r) => Self::UnknownNumber(l + r),
            },
        }
    }
}
impl Sub for Constant {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Self::UInt(if l { 1 } else { 0 } - if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(if l { 1 } else { 0 } - r),
                Self::SInt(r) => Self::SInt(if l { 1 } else { 0 } - r),
                Self::Float(r) => Self::Float(if l { 1.0 } else { 0.0 } - r),
                Self::UnknownInt(r) => Self::UnknownInt(if l { 1 } else { 0 } - r),
                Self::UnknownNumber(r) => Self::UnknownNumber(if l { 1.0 } else { 0.0 } - r),
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Self::UInt(l - if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(l - r),
                Self::SInt(r) => Self::SInt(l as i32 - r),
                Self::Float(r) => Self::Float(l as f32 - r),
                Self::UnknownInt(r) => Self::UInt(l - r as u32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 - r),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l - if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l - r as i32),
                Self::SInt(r) => Self::SInt(l - r),
                Self::Float(r) => Self::Float(l as f32 - r),
                Self::UnknownInt(r) => Self::SInt(l - r as i32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 - r),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Self::Float(l - if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l - r as f32),
                Self::SInt(r) => Self::Float(l - r as f32),
                Self::Float(r) => Self::Float(l - r),
                Self::UnknownInt(r) => Self::Float(l - r as f32),
                Self::UnknownNumber(r) => Self::Float(l - r as f32),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l as i32 - if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l as i32 - r as i32),
                Self::SInt(r) => Self::SInt(l as i32 - r),
                Self::Float(r) => Self::Float(l as f32 - r),
                Self::UnknownInt(r) => Self::UnknownInt(l - r),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 - r),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Self::Float(l as f32 - if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l as f32 - r as f32),
                Self::SInt(r) => Self::Float(l as f32 - r as f32),
                Self::Float(r) => Self::Float(l as f32 - r),
                Self::UnknownInt(r) => Self::UnknownNumber(l - r as f64),
                Self::UnknownNumber(r) => Self::UnknownNumber(l - r),
            },
        }
    }
}
impl Mul for Constant {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Self::UInt(if l { 1 } else { 0 } * if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(if l { 1 } else { 0 } * r),
                Self::SInt(r) => Self::SInt(if l { 1 } else { 0 } * r),
                Self::Float(r) => Self::Float(if l { 1.0 } else { 0.0 } * r),
                Self::UnknownInt(r) => Self::UnknownInt(if l { 1 } else { 0 } * r),
                Self::UnknownNumber(r) => Self::UnknownNumber(if l { 1.0 } else { 0.0 } * r),
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Self::UInt(l * if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(l * r),
                Self::SInt(r) => Self::SInt(l as i32 * r),
                Self::Float(r) => Self::Float(l as f32 * r),
                Self::UnknownInt(r) => Self::UInt(l * r as u32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 * r),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l * if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l * r as i32),
                Self::SInt(r) => Self::SInt(l * r),
                Self::Float(r) => Self::Float(l as f32 * r),
                Self::UnknownInt(r) => Self::SInt(l * r as i32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 * r),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Self::Float(l * if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l * r as f32),
                Self::SInt(r) => Self::Float(l * r as f32),
                Self::Float(r) => Self::Float(l * r),
                Self::UnknownInt(r) => Self::Float(l * r as f32),
                Self::UnknownNumber(r) => Self::Float(l * r as f32),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l as i32 * if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l as i32 * r as i32),
                Self::SInt(r) => Self::SInt(l as i32 * r),
                Self::Float(r) => Self::Float(l as f32 * r),
                Self::UnknownInt(r) => Self::UnknownInt(l * r),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 * r),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Self::Float(l as f32 * if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l as f32 * r as f32),
                Self::SInt(r) => Self::Float(l as f32 * r as f32),
                Self::Float(r) => Self::Float(l as f32 * r),
                Self::UnknownInt(r) => Self::UnknownNumber(l * r as f64),
                Self::UnknownNumber(r) => Self::UnknownNumber(l * r),
            },
        }
    }
}
impl Div for Constant {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Self::UInt(if l { 1 } else { 0 } / if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(if l { 1 } else { 0 } / r),
                Self::SInt(r) => Self::SInt(if l { 1 } else { 0 } / r),
                Self::Float(r) => Self::Float(if l { 1.0 } else { 0.0 } / r),
                Self::UnknownInt(r) => Self::UnknownInt(if l { 1 } else { 0 } / r),
                Self::UnknownNumber(r) => Self::UnknownNumber(if l { 1.0 } else { 0.0 } / r),
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Self::UInt(l / if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(l / r),
                Self::SInt(r) => Self::SInt(l as i32 / r),
                Self::Float(r) => Self::Float(l as f32 / r),
                Self::UnknownInt(r) => Self::UInt(l / r as u32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 / r),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l / if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l / r as i32),
                Self::SInt(r) => Self::SInt(l / r),
                Self::Float(r) => Self::Float(l as f32 / r),
                Self::UnknownInt(r) => Self::SInt(l / r as i32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 / r),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Self::Float(l / if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l / r as f32),
                Self::SInt(r) => Self::Float(l / r as f32),
                Self::Float(r) => Self::Float(l / r),
                Self::UnknownInt(r) => Self::Float(l / r as f32),
                Self::UnknownNumber(r) => Self::Float(l / r as f32),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l as i32 / if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l as i32 / r as i32),
                Self::SInt(r) => Self::SInt(l as i32 / r),
                Self::Float(r) => Self::Float(l as f32 / r),
                Self::UnknownInt(r) => Self::UnknownInt(l / r),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 / r),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Self::Float(l as f32 / if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l as f32 / r as f32),
                Self::SInt(r) => Self::Float(l as f32 / r as f32),
                Self::Float(r) => Self::Float(l as f32 / r),
                Self::UnknownInt(r) => Self::UnknownNumber(l / r as f64),
                Self::UnknownNumber(r) => Self::UnknownNumber(l / r),
            },
        }
    }
}
impl Rem for Constant {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Self::UInt(if l { 1 } else { 0 } % if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(if l { 1 } else { 0 } % r),
                Self::SInt(r) => Self::SInt(if l { 1 } else { 0 } % r),
                Self::Float(r) => Self::Float(if l { 1.0 } else { 0.0 } % r),
                Self::UnknownInt(r) => Self::UnknownInt(if l { 1 } else { 0 } % r),
                Self::UnknownNumber(r) => Self::UnknownNumber(if l { 1.0 } else { 0.0 } % r),
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Self::UInt(l % if r { 1 } else { 0 }),
                Self::UInt(r) => Self::UInt(l % r),
                Self::SInt(r) => Self::SInt(l as i32 % r),
                Self::Float(r) => Self::Float(l as f32 % r),
                Self::UnknownInt(r) => Self::UInt(l % r as u32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 % r),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l % if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l % r as i32),
                Self::SInt(r) => Self::SInt(l % r),
                Self::Float(r) => Self::Float(l as f32 % r),
                Self::UnknownInt(r) => Self::SInt(l % r as i32),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 % r),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Self::Float(l % if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l % r as f32),
                Self::SInt(r) => Self::Float(l % r as f32),
                Self::Float(r) => Self::Float(l % r),
                Self::UnknownInt(r) => Self::Float(l % r as f32),
                Self::UnknownNumber(r) => Self::Float(l % r as f32),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Self::SInt(l as i32 % if r { 1 } else { 0 }),
                Self::UInt(r) => Self::SInt(l as i32 % r as i32),
                Self::SInt(r) => Self::SInt(l as i32 % r),
                Self::Float(r) => Self::Float(l as f32 % r),
                Self::UnknownInt(r) => Self::UnknownInt(l % r),
                Self::UnknownNumber(r) => Self::UnknownNumber(l as f64 % r),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Self::Float(l as f32 % if r { 1.0 } else { 0.0 }),
                Self::UInt(r) => Self::Float(l as f32 % r as f32),
                Self::SInt(r) => Self::Float(l as f32 % r as f32),
                Self::Float(r) => Self::Float(l as f32 % r),
                Self::UnknownInt(r) => Self::UnknownNumber(l % r as f64),
                Self::UnknownNumber(r) => Self::UnknownNumber(l % r),
            },
        }
    }
}
impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match self {
            &Self::Bool(l) => match other {
                &Self::Bool(r) => l == r,
                &Self::UInt(r) => {
                    if l {
                        r == 1
                    } else {
                        r == 0
                    }
                }
                &Self::SInt(r) => {
                    if l {
                        r == 1
                    } else {
                        r == 0
                    }
                }
                &Self::Float(r) => {
                    if l {
                        r == 1.0
                    } else {
                        r == 0.0
                    }
                }
                &Self::UnknownInt(r) => {
                    if l {
                        r == 1
                    } else {
                        r == 0
                    }
                }
                &Self::UnknownNumber(r) => {
                    if l {
                        r == 1.0
                    } else {
                        r == 0.0
                    }
                }
            },
            &Self::UInt(l) => match other {
                &Self::Bool(r) => {
                    if r {
                        l == 1
                    } else {
                        l == 0
                    }
                }
                &Self::UInt(r) => l == r,
                &Self::SInt(r) => l as i64 == r as i64,
                &Self::Float(r) => l as f32 == r,
                &Self::UnknownInt(r) => l as isize == r,
                &Self::UnknownNumber(r) => l as f64 == r,
            },
            &Self::SInt(l) => match other {
                Self::Bool(true) => l == 1,
                Self::Bool(false) => l == 0,
                &Self::UInt(r) => l as i64 == r as i64,
                &Self::SInt(r) => l == r,
                &Self::Float(r) => l as f32 == r,
                &Self::UnknownInt(r) => l as isize == r,
                &Self::UnknownNumber(r) => l as f64 == r,
            },
            &Self::Float(l) => match other {
                Self::Bool(true) => l == 1.0,
                Self::Bool(false) => l == 0.0,
                &Self::UInt(r) => l == r as f32,
                &Self::SInt(r) => l == r as f32,
                &Self::Float(r) => l == r,
                &Self::UnknownInt(r) => l == r as f32,
                &Self::UnknownNumber(r) => l as f64 == r,
            },
            &Self::UnknownInt(l) => match other {
                Self::Bool(true) => l == 1,
                Self::Bool(false) => l == 0,
                &Self::UInt(r) => l == r as isize,
                &Self::SInt(r) => l == r as isize,
                &Self::Float(r) => l as f32 == r,
                &Self::UnknownInt(r) => l == r,
                &Self::UnknownNumber(r) => l as f64 == r,
            },
            &Self::UnknownNumber(l) => match other {
                Self::Bool(true) => l == 1.0,
                Self::Bool(false) => l == 0.0,
                &Self::UInt(r) => l == r as f64,
                &Self::SInt(r) => l == r as f64,
                &Self::Float(r) => l == r as f64,
                &Self::UnknownInt(r) => l == r as f64,
                &Self::UnknownNumber(r) => l == r,
            },
        }
    }
}
impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self {
            Self::Bool(true) => match other {
                Self::Bool(true) => Some(std::cmp::Ordering::Equal),
                Self::Bool(false) => Some(std::cmp::Ordering::Greater),
                Self::UInt(r) => 1.partial_cmp(r),
                Self::SInt(r) => 1.partial_cmp(r),
                Self::Float(r) => 1.0.partial_cmp(r),
                Self::UnknownInt(r) => 1.partial_cmp(r),
                Self::UnknownNumber(r) => 1.0.partial_cmp(r),
            },
            Self::Bool(false) => match other {
                Self::Bool(true) => Some(std::cmp::Ordering::Less),
                Self::Bool(false) => Some(std::cmp::Ordering::Equal),
                Self::UInt(r) => 0.partial_cmp(r),
                Self::SInt(r) => 0.partial_cmp(r),
                Self::Float(r) => 0.0.partial_cmp(r),
                Self::UnknownInt(r) => 0.partial_cmp(r),
                Self::UnknownNumber(r) => 0.0.partial_cmp(r),
            },
            &Self::UInt(l) => match other {
                Self::Bool(true) => l.partial_cmp(&1),
                Self::Bool(false) => l.partial_cmp(&0),
                Self::UInt(r) => l.partial_cmp(r),
                Self::SInt(r) => (l as i64).partial_cmp(&(*r as i64)),
                Self::Float(r) => (l as f32).partial_cmp(r),
                Self::UnknownInt(r) => (l as isize).partial_cmp(r),
                Self::UnknownNumber(r) => (l as f64).partial_cmp(r),
            },
            &Self::SInt(l) => match other {
                Self::Bool(true) => l.partial_cmp(&1),
                Self::Bool(false) => l.partial_cmp(&0),
                Self::UInt(r) => (l as i64).partial_cmp(&(*r as i64)),
                Self::SInt(r) => l.partial_cmp(r),
                Self::Float(r) => (l as f32).partial_cmp(r),
                Self::UnknownInt(r) => (l as isize).partial_cmp(r),
                Self::UnknownNumber(r) => (l as f64).partial_cmp(r),
            },
            &Self::Float(l) => match other {
                Self::Bool(true) => l.partial_cmp(&1.0),
                Self::Bool(false) => l.partial_cmp(&0.0),
                Self::UInt(r) => l.partial_cmp(&(*r as f32)),
                Self::SInt(r) => l.partial_cmp(&(*r as f32)),
                Self::Float(r) => l.partial_cmp(r),
                Self::UnknownInt(r) => l.partial_cmp(&(*r as f32)),
                Self::UnknownNumber(r) => (l as f64).partial_cmp(r),
            },
            &Self::UnknownInt(l) => match other {
                Self::Bool(true) => l.partial_cmp(&1),
                Self::Bool(false) => l.partial_cmp(&0),
                Self::UInt(r) => l.partial_cmp(&(*r as isize)),
                Self::SInt(r) => l.partial_cmp(&(*r as isize)),
                Self::Float(r) => (l as f32).partial_cmp(r),
                Self::UnknownInt(r) => l.partial_cmp(r),
                Self::UnknownNumber(r) => (l as f64).partial_cmp(r),
            },
            &Self::UnknownNumber(l) => match other {
                Self::Bool(true) => l.partial_cmp(&1.0),
                Self::Bool(false) => l.partial_cmp(&0.0),
                Self::UInt(r) => l.partial_cmp(&(*r as f64)),
                Self::SInt(r) => l.partial_cmp(&(*r as f64)),
                Self::Float(r) => l.partial_cmp(&(*r as f64)),
                Self::UnknownInt(r) => l.partial_cmp(&(*r as f64)),
                Self::UnknownNumber(r) => l.partial_cmp(r),
            },
        }
    }
}
impl Constant {
    pub fn try_pow(self, rhs: Self) -> Option<Self> {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(if l { 1u32 } else { 0u32 }.pow(if r {
                    1
                } else {
                    0
                }))),
                Self::UInt(r) => Some(Self::UInt(if l { 1u32 } else { 0u32 }.pow(r))),
                Self::SInt(r) if r >= 0 => {
                    Some(Self::SInt(if l { 1i32 } else { 0i32 }.pow(r as _)))
                }
                Self::SInt(r) => Some(Self::Float(if l { 1.0f32 } else { 0.0f32 }.powi(r))),
                Self::Float(r) => Some(Self::Float(if l { 1.0f32 } else { 0.0f32 }.powf(r))),
                Self::UnknownInt(r) if r >= 0 => Some(Self::UnknownInt(
                    if l { 1isize } else { 0isize }.pow(r as _),
                )),
                Self::UnknownInt(r) => Some(Self::UnknownNumber(
                    if l { 1.0f64 } else { 0.0f64 }.powi(r as _),
                )),
                Self::UnknownNumber(r) => {
                    Some(Self::UnknownNumber(if l { 1.0f64 } else { 0.0f64 }.powf(r)))
                }
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(l.pow(if r { 1 } else { 0 }))),
                Self::UInt(r) => Some(Self::UInt(l.pow(r))),
                Self::SInt(r) if r >= 0 => Some(Self::SInt((l as i32).pow(r as _))),
                Self::SInt(r) => None,
                Self::Float(r) => Some(Self::Float((l as f32).powf(r))),
                Self::UnknownInt(r) if r >= 0 => Some(Self::UInt(l.pow(r as u32))),
                Self::UnknownInt(r) => None,
                Self::UnknownNumber(r) => Some(Self::UnknownNumber((l as f64).powf(r))),
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l.pow(if r { 1 } else { 0 }))),
                Self::UInt(r) => Some(Self::SInt(l.pow(r))),
                Self::SInt(r) if r >= 0 => Some(Self::SInt(l.pow(r as _))),
                Self::SInt(_) => None,
                Self::Float(r) => Some(Self::Float((l as f32).powf(r))),
                Self::UnknownInt(r) if r >= 0 => Some(Self::SInt(l.pow(r as _))),
                Self::UnknownInt(_) => None,
                Self::UnknownNumber(r) => Some(Self::UnknownNumber((l as f64).powf(r))),
            },
            Self::Float(l) => match rhs {
                Self::Bool(r) => Some(Self::Float(l.powi(if r { 1 } else { 0 }))),
                Self::UInt(r) => Some(Self::Float(l.powi(r as _))),
                Self::SInt(r) => Some(Self::Float(l.powi(r))),
                Self::Float(r) => Some(Self::Float(l.powf(r))),
                Self::UnknownInt(r) => Some(Self::Float(l.powi(r as _))),
                Self::UnknownNumber(r) => Some(Self::Float(l.powf(r as _))),
            },
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt((l as i32).pow(if r { 1 } else { 0 }))),
                Self::UInt(r) => Some(Self::SInt((l as i32).pow(r))),
                Self::SInt(r) if r >= 0 => Some(Self::SInt((l as i32).pow(r as _))),
                Self::SInt(_) => None,
                Self::Float(r) => Some(Self::Float((l as f32).powf(r))),
                Self::UnknownInt(r) if r >= 0 => Some(Self::UnknownInt(l.pow(r as _))),
                Self::UnknownInt(_) => None,
                Self::UnknownNumber(r) => Some(Self::UnknownNumber((l as f64).powf(r))),
            },
            Self::UnknownNumber(l) => match rhs {
                Self::Bool(r) => Some(Self::Float((l as f32).powi(if r { 1 } else { 0 }))),
                Self::UInt(r) => Some(Self::Float((l as f32).powi(r as _))),
                Self::SInt(r) => Some(Self::Float((l as f32).powi(r))),
                Self::Float(r) => Some(Self::Float((l as f32).powf(r as _))),
                Self::UnknownInt(r) => Some(Self::UnknownNumber(l.powi(r as _))),
                Self::UnknownNumber(r) => Some(Self::UnknownNumber(l.powf(r))),
            },
        }
    }

    pub const fn try_bit_and(self, rhs: Self) -> Option<Self> {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(if l { 1 } else { 0 } & if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(if l { 1 } else { 0 } & r)),
                Self::SInt(r) => Some(Self::SInt(if l { 1 } else { 0 } & r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(if l { 1 } else { 0 } & r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(l & if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(l & r)),
                Self::SInt(r) => Some(Self::SInt(l as i32 & r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UInt(l & r as u32)),
                Self::UnknownNumber(_) => None,
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l & if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l & r as i32)),
                Self::SInt(r) => Some(Self::SInt(l & r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::SInt(l & r as i32)),
                Self::UnknownNumber(_) => None,
            },
            Self::Float(_) => None,
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l as i32 & if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l as i32 & r as i32)),
                Self::SInt(r) => Some(Self::SInt(l as i32 & r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(l & r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UnknownNumber(_) => None,
        }
    }

    pub const fn try_bit_or(self, rhs: Self) -> Option<Self> {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(if l { 1 } else { 0 } | if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(if l { 1 } else { 0 } | r)),
                Self::SInt(r) => Some(Self::SInt(if l { 1 } else { 0 } | r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(if l { 1 } else { 0 } | r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(l | if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(l | r)),
                Self::SInt(r) => Some(Self::SInt(l as i32 | r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UInt(l | r as u32)),
                Self::UnknownNumber(_) => None,
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l | if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l | r as i32)),
                Self::SInt(r) => Some(Self::SInt(l | r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::SInt(l | r as i32)),
                Self::UnknownNumber(_) => None,
            },
            Self::Float(_) => None,
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l as i32 | if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l as i32 | r as i32)),
                Self::SInt(r) => Some(Self::SInt(l as i32 | r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(l | r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UnknownNumber(_) => None,
        }
    }

    pub const fn try_bit_xor(self, rhs: Self) -> Option<Self> {
        match self {
            Self::Bool(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(if l { 1 } else { 0 } ^ if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(if l { 1 } else { 0 } ^ r)),
                Self::SInt(r) => Some(Self::SInt(if l { 1 } else { 0 } ^ r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(if l { 1 } else { 0 } ^ r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UInt(l) => match rhs {
                Self::Bool(r) => Some(Self::UInt(l ^ if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::UInt(l ^ r)),
                Self::SInt(r) => Some(Self::SInt(l as i32 ^ r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UInt(l ^ r as u32)),
                Self::UnknownNumber(_) => None,
            },
            Self::SInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l ^ if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l ^ r as i32)),
                Self::SInt(r) => Some(Self::SInt(l ^ r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::SInt(l ^ r as i32)),
                Self::UnknownNumber(_) => None,
            },
            Self::Float(_) => None,
            Self::UnknownInt(l) => match rhs {
                Self::Bool(r) => Some(Self::SInt(l as i32 ^ if r { 1 } else { 0 })),
                Self::UInt(r) => Some(Self::SInt(l as i32 ^ r as i32)),
                Self::SInt(r) => Some(Self::SInt(l as i32 ^ r)),
                Self::Float(_) => None,
                Self::UnknownInt(r) => Some(Self::UnknownInt(l ^ r)),
                Self::UnknownNumber(_) => None,
            },
            Self::UnknownNumber(_) => None,
        }
    }

    pub const fn logical_and(self, rhs: Self) -> bool {
        self.into_bool() && rhs.into_bool()
    }

    pub const fn logical_or(self, rhs: Self) -> bool {
        self.into_bool() || rhs.into_bool()
    }
}
