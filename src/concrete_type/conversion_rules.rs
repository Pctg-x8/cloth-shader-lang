//! 自動型変換ルールまわり

use super::{
    ConcreteType, IntrinsicMatrixType, IntrinsicScalarType, IntrinsicType, IntrinsicVectorType,
};

/// 演算子の項の値変換指示
pub enum BinaryOpScalarConversion {
    /// 変換なし
    None,
    /// UnknownNumberClassに昇格（主にUnknownIntNumberに対して）
    PromoteUnknownNumber,
    /// 指定した型にキャスト
    Cast(IntrinsicScalarType),
    /// 指定した型にUnknownHogehogeClassの値を実体化する
    Instantiate(IntrinsicScalarType),
}

/// 分配オペレーションの必要性を表す
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpValueDistributionRequirements {
    /// 左の項の分配が必要
    LeftTerm,
    /// 右の項の分配が必要
    RightTerm,
}

/// 二項演算の型変換/分配指示データ
pub struct BinaryOpTypeConversion2<'s> {
    /// 演算子の左の項の変換指示
    pub left_op: BinaryOpScalarConversion,
    /// 演算子の右の項の変換指示
    pub right_op: BinaryOpScalarConversion,
    /// どちらの項の値の分配すべきか Noneの場合は分配処理はなし
    pub dist: Option<BinaryOpValueDistributionRequirements>,
    /// この演算の最終的な型
    pub result_type: ConcreteType<'s>,
}

/// （乗算以外の）組み込み算術演算の自動型変換ルールの定義
///
/// ベクトル/行列との乗算に特殊なルールが存在するので、乗算は別で定義
pub fn arithmetic_compare_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ型

        if (lhs.scalar_type(), rhs.scalar_type())
            == (
                Some(IntrinsicScalarType::Bool),
                Some(IntrinsicScalarType::Bool),
            )
        {
            // Bool型どうしの算術演算はSIntに上げる
            return Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
                right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
                dist: None,
                result_type: lhs
                    .clone()
                    .try_cast_intrinsic_scalar(IntrinsicScalarType::SInt)?,
            });
        }

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()?,
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // 片方でもUnitの場合は変換なし（他のところで形マッチエラーにさせる）
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // boolの算術演算はSIntとして行う（型一致時の特殊ルール）
        (IntrinsicScalarType::Bool, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        // 同じ型だったら変換なし（この場合はDistributionだけが走る）
        (IntrinsicScalarType::UInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // UnknownHogehogeClassは適当にInstantiateさせて、あとはBoolとの演算ルールに従う
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::SInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::SInt),
            IntrinsicScalarType::SInt,
        ),
        // 左の型に合わせるパターン
        (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 左の型に合わせるパターン（Instantiate）
        (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
        // 左の型に合わせるパターン（昇格）
        (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::PromoteUnknownNumber,
            l,
        ),
        // 右の型に合わせるパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右の型に合わせるパターン（Instantiate）
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右の型に合わせるパターン（昇格）
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::PromoteUnknownNumber,
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右をInstantiateしたうえで、すべての演算をFloatで行うパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            IntrinsicScalarType::Float,
        ),
        // 左をInstantiateしたうえで、すべての演算をFloatで行うパターン
        (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
            IntrinsicScalarType::Float,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

/// 組み込みビット演算の自動型変換ルールの定義
pub fn bitwise_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ型

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()?,
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // 片方でもUnitの場合は変換なし（他のところで形マッチエラーにさせる）
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // 片方でもFloatの場合は演算不可 UnknownNumberClassも同様
        (IntrinsicScalarType::Float, _)
        | (_, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownNumberClass, _)
        | (_, IntrinsicScalarType::UnknownNumberClass) => return None,
        // 同じ型だったら変換なし（この場合はDistributionだけが走る）
        (IntrinsicScalarType::Bool, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // UnknownHogehogeClassは適当にInstantiateさせて、あとはBoolとの演算ルールに従う
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
            IntrinsicScalarType::UInt,
        ),
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
            BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
            IntrinsicScalarType::UInt,
        ),
        // 右に合わせるパターン
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 左に合わせるパターン
        (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 左をInstantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
        | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 右をInstantiateするパターン
        (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

/// 組み込み論理積（`&&`）/論理和（`||`）の自動型変換ルール定義
pub fn logical_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()?,
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // Unitとの演算は不可能
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // 同型
        (IntrinsicScalarType::Bool, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // 同型だがBoolにInstantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Bool),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Bool),
            IntrinsicScalarType::Bool,
        ),
        // Instantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
        // 左に合わせるパターン（汎用）
        (IntrinsicScalarType::Bool, _) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 右に合わせるパターン（汎用）
        (_, IntrinsicScalarType::Bool) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 両方ともBoolにキャスト/Instantiateするパターン（汎用）
        (l, r) => (
            if l.is_unknown_type() {
                BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Bool)
            } else {
                BinaryOpScalarConversion::Cast(IntrinsicScalarType::Bool)
            },
            if r.is_unknown_type() {
                BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Bool)
            } else {
                BinaryOpScalarConversion::Cast(IntrinsicScalarType::Bool)
            },
            IntrinsicScalarType::Bool,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

/// 組み込み累乗（pow）の自動型変換ルール
pub fn pow_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    // 小ディメンションの値をより広いディメンションの値へと自動変換する（1.0 -> Float4(1.0, 1.0, 1.0, 1.0)みたいに値を分配する（distribute）操作）
    let (dist, composite_type_class) = match (lhs, rhs) {
        // 右がでかいので右に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::LeftTerm),
            rhs.intrinsic_composite_type_class()?,
        ),
        // 左がでかいので左に合わせるパターン
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(_)),
        )
        | (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(_)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(_)),
        ) => (
            Some(BinaryOpValueDistributionRequirements::RightTerm),
            lhs.intrinsic_composite_type_class()?,
        ),
        // 同じサイズ
        _ => (None, lhs.intrinsic_composite_type_class()?),
    };

    // 自動型変換ルール
    let (l, r) = (lhs.scalar_type()?, rhs.scalar_type()?);
    let (left_conv, right_conv, result_type) = match (l, r) {
        // Unitとの演算は不可能
        (IntrinsicScalarType::Unit, _) | (_, IntrinsicScalarType::Unit) => return None,
        // 同型
        (IntrinsicScalarType::Float, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::None,
            l,
        ),
        // 同型だがFloatにInstantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
            IntrinsicScalarType::Float,
        ),
        // Instantiateするパターン
        (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float)
        | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Instantiate(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass)
        | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Instantiate(l),
            l,
        ),
        // 左に合わせるパターン（汎用）
        (IntrinsicScalarType::Float, _) => (
            BinaryOpScalarConversion::None,
            BinaryOpScalarConversion::Cast(l),
            l,
        ),
        // 右に合わせるパターン（汎用）
        (_, IntrinsicScalarType::Float) => (
            BinaryOpScalarConversion::Cast(r),
            BinaryOpScalarConversion::None,
            r,
        ),
        // 両方ともFloatにキャスト/Instantiateするパターン（汎用）
        (l, r) => (
            if l.is_unknown_type() {
                BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
            } else {
                BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
            },
            if r.is_unknown_type() {
                BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
            } else {
                BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
            },
            IntrinsicScalarType::Float,
        ),
    };

    Some(BinaryOpTypeConversion2 {
        left_op: left_conv,
        right_op: right_conv,
        dist,
        result_type: ConcreteType::Intrinsic(composite_type_class.combine_scalar(result_type)),
    })
}

/// 乗算の自動型変換ルールの定義
pub fn multiply_op_type_conversion<'s>(
    lhs: &ConcreteType<'s>,
    rhs: &ConcreteType<'s>,
) -> Option<BinaryOpTypeConversion2<'s>> {
    if lhs == rhs {
        // 同じ型

        return Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: lhs.clone(),
        });
    }

    match (lhs, rhs) {
        // vector times scalar
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(a, _))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(b)),
        ) if a == b => {
            // both same element type and scalar type
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: BinaryOpScalarConversion::None,
                dist: None,
                result_type: lhs.clone(),
            })
        }
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(
                IntrinsicScalarType::Float,
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(r)),
        ) => {
            // force casting/instantiating right hand to float
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: if r.is_unknown_type() {
                    BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
                } else {
                    BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
                },
                dist: None,
                result_type: lhs.clone(),
            })
        }
        // matrix times scalar
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(a, _),
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(b)),
        ) if a == b => {
            // both same element type and scalar type
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: BinaryOpScalarConversion::None,
                dist: None,
                result_type: lhs.clone(),
            })
        }
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(IntrinsicScalarType::Float, _),
                _,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Scalar(r)),
        ) => {
            // force casting/instantiating right hand to float
            Some(BinaryOpTypeConversion2 {
                left_op: BinaryOpScalarConversion::None,
                right_op: if r.is_unknown_type() {
                    BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float)
                } else {
                    BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float)
                },
                dist: None,
                result_type: lhs.clone(),
            })
        }
        // vector times matrix
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(rl)),
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(rr, c))),
        ) if rl == rr => Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: IntrinsicType::Vector(IntrinsicVectorType(rl.0, *c)).into(),
        }),
        // matrix times vector
        (
            ConcreteType::Intrinsic(IntrinsicType::Matrix(IntrinsicMatrixType(
                IntrinsicVectorType(IntrinsicScalarType::Float, r),
                cl,
            ))),
            ConcreteType::Intrinsic(IntrinsicType::Vector(IntrinsicVectorType(
                IntrinsicScalarType::Float,
                rr,
            ))),
        ) if cl == rr => Some(BinaryOpTypeConversion2 {
            left_op: BinaryOpScalarConversion::None,
            right_op: BinaryOpScalarConversion::None,
            dist: None,
            result_type: IntrinsicType::vec(*r).into(),
        }),
        // between same length vectors
        (
            ConcreteType::Intrinsic(IntrinsicType::Vector(vl)),
            ConcreteType::Intrinsic(IntrinsicType::Vector(vr)),
        ) if vl.len() == vr.len() => {
            match (vl.scalar(), vr.scalar()) {
                // simple casting
                // empowered right hand
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::Bool, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::Float) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(vr.0),
                        right_op: BinaryOpScalarConversion::None,
                        dist: None,
                        result_type: rhs.clone(),
                    })
                }
                // empowered left hand
                (IntrinsicScalarType::Float, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::None,
                        right_op: BinaryOpScalarConversion::Cast(vl.0),
                        dist: None,
                        result_type: lhs.clone(),
                    })
                }
                // instantiate lhs, cast rhs, operate on uint type
                (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
                        right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
                        dist: None,
                        result_type: IntrinsicType::Vector(vl.with_type(IntrinsicScalarType::UInt))
                            .into(),
                    })
                }
                // instantiate rhs, cast lhs, operate on uint type
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::UInt),
                        right_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::UInt),
                        dist: None,
                        result_type: IntrinsicType::Vector(vl.with_type(IntrinsicScalarType::UInt))
                            .into(),
                    })
                }
                // simply instantiate to rhs type
                (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt)
                | (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(vr.0),
                        right_op: BinaryOpScalarConversion::None,
                        dist: None,
                        result_type: rhs.clone(),
                    })
                }
                // simply instantiate to lhs type
                (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass)
                | (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::None,
                        right_op: BinaryOpScalarConversion::Instantiate(vl.0),
                        dist: None,
                        result_type: lhs.clone(),
                    })
                }
                // instantiate lhs to float, cast rhs to float
                (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt)
                | (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
                        right_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
                        dist: None,
                        result_type: IntrinsicType::Vector(
                            vl.with_type(IntrinsicScalarType::Float),
                        )
                        .into(),
                    })
                }
                // instantiate rhs to float, cast lhs to float
                (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass)
                | (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass)
                | (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => {
                    Some(BinaryOpTypeConversion2 {
                        left_op: BinaryOpScalarConversion::Cast(IntrinsicScalarType::Float),
                        right_op: BinaryOpScalarConversion::Instantiate(IntrinsicScalarType::Float),
                        dist: None,
                        result_type: IntrinsicType::Vector(
                            vl.with_type(IntrinsicScalarType::Float),
                        )
                        .into(),
                    })
                }
                // never
                _ => None,
            }
        }
        // never
        _ => None,
    }
}
