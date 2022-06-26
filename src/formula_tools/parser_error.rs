use core::fmt;

/// Result with internal [`Error`] type.
pub type Result<T> = core::result::Result<T, ParserError>;

/// Error type.
#[derive(Debug, Eq, PartialEq)]
pub enum ParserError {
    // A charge was present without being attached to a chemical formula.
    ChargeNoFormula,
    // The formula contained an invalid charge coefficient.
    InvalidChargeCoefficient,
    /// A charge may only occur at the end of the formula.
    InvalidChargePosition,
    /// The formula contained an invalid coefficient.
    InvalidCoefficient,
    /// The formula contained a mismatched parenthesis.
    MismatchedParenthesis,
    /// The formula contained a segment coefficient without a segment.
    CoefficientNoSegment,
    /// The formula contained an unrecognized (probably invalid) elemental symbol.
    UnrecognizedSymbol,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ParserError::ChargeNoFormula => "A charge was present without a formula being present.",
            ParserError::InvalidChargeCoefficient => {
                "The formula contained an invalid charge coefficient."
            }
            ParserError::InvalidChargePosition => {
                "A charge may only occur at the end of the formula."
            }
            ParserError::InvalidCoefficient => "The formula contained an invalid coefficient",
            ParserError::MismatchedParenthesis => "The formula contained a mismatched parenthesis.",
            ParserError::CoefficientNoSegment => {
                "A segment coefficient was present, but without a segment."
            }
            ParserError::UnrecognizedSymbol => {
                "The formula contained an unrecognized elemental symbol."
            }
        })
    }
}

impl std::error::Error for ParserError {}
