use core::fmt;

/// Result with internal [`Error`] type.
pub type Result<T> = core::result::Result<T, ParserError>;

/// Error type.
#[derive(Debug, Eq, PartialEq)]
pub enum ParserError {
    /// The formula contained an empty segment.
    EmptySegment,
    // The formula contained an invalid charge.
    InvalidCharge,
    /// The formula contained an invalid multiplier.
    InvalidMultiplier,
    /// The formula contained a mismatched parenthesis.
    MismatchedParenthesis,
    /// The formula contained a segment multiplier without a segment.
    MultiplierNoSegment,
    /// A mid-dot was present in the formula, but is unsupported by the parser.
    UnexpectedMidDot,
    /// The formula contained an unrecognized (probably invalid) elemental symbol.
    UnrecognizedSymbol,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ParserError::EmptySegment => "The formula contained an empty segment.",
            ParserError::InvalidCharge => "The formula contained an invalid charge",
            ParserError::InvalidMultiplier => "The formula contained an invalid multiplier",
            ParserError::MismatchedParenthesis => "The formula contained a mismatched parenthesis.",
            ParserError::MultiplierNoSegment => {
                "A segment multiplier was present, but without a segment."
            }
            ParserError::UnexpectedMidDot => {
                "A mid-dot segmented formula is unsupported by the parser."
            }
            ParserError::UnrecognizedSymbol => {
                "The formula contained an unrecognized elemental symbol."
            }
        })
    }
}

impl std::error::Error for ParserError {}
