use core::fmt;

/// Result with internal [`Error`] type.
pub type Result<T> = core::result::Result<T, ParserError>;

/// Error type.
#[derive(Debug, Eq, PartialEq)]
pub enum ParserError {
    // A charge was present without being attached to a chemical formula.
    ChargeNoFormula,
    // The formula contained an invalid charge multiplier.
    InvalidChargeMultiplier,
    /// A charge may only occur at the end of the formula.
    InvalidChargePosition,
    /// The formula contained an invalid multiplier.
    InvalidMultiplier,
    /// The formula contained a mismatched parenthesis.
    MismatchedParenthesis,
    /// The formula contained a segment multiplier without a segment.
    MultiplierNoSegment,
    /// The formula contained an unrecognized (probably invalid) elemental symbol.
    UnrecognizedSymbol,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ParserError::ChargeNoFormula => "A charge was present without a formula being present.",
            ParserError::InvalidChargeMultiplier => {
                "The formula contained an invalid charge multiplier."
            }
            ParserError::InvalidChargePosition => {
                "A charge may only occur at the end of the formula."
            }
            ParserError::InvalidMultiplier => "The formula contained an invalid multiplier",
            ParserError::MismatchedParenthesis => "The formula contained a mismatched parenthesis.",
            ParserError::MultiplierNoSegment => {
                "A segment multiplier was present, but without a segment."
            }
            ParserError::UnrecognizedSymbol => {
                "The formula contained an unrecognized elemental symbol."
            }
        })
    }
}

impl std::error::Error for ParserError {}
