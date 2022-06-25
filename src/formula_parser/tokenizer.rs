#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenTypes {
    /// Initial token.
    Start,
    /// A numeric character.
    Digit(char),
    /// A left (opening) parenthesis.
    LParen,
    /// A right (closing) parenthesis.
    RParen,
    /// The characters of an element symbol.
    ElementHead(char),
    /// The characters of an element symbol.
    ElementTail(char),
    /// A mid-dot special character.
    ChargeSign(char),
    /// The end token.
    End,
}

pub fn tokenize_string(chars: &[char]) -> Vec<TokenTypes> {
    let mut tokens: Vec<TokenTypes> = vec![TokenTypes::Start];

    for (i, c) in chars.iter().enumerate() {
        match c {
            '(' => {
                tokens.push(TokenTypes::LParen);
            }
            ')' => {
                tokens.push(TokenTypes::RParen);
            }
            '0'..='9' => {
                tokens.push(TokenTypes::Digit(*c));
            }
            'A'..='Z' => {
                tokens.push(TokenTypes::ElementHead(*c));
            }
            'a'..='z' => {
                tokens.push(TokenTypes::ElementTail(*c));
            }
            '+' => {
                tokens.push(TokenTypes::ChargeSign('+'));
            }
            '-' => {
                tokens.push(TokenTypes::ChargeSign('-'));
            }
            _ => {
                eprintln!("Invalid character, {}, at index {}", c, i);
            }
        }
    }

    tokens.push(TokenTypes::End);

    tokens
}
