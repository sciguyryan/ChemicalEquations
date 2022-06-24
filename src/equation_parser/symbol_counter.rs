use std::ops::MulAssign;

#[derive(Debug, Clone)]
pub struct SymbolCounter {
    pub symbol: String,
    pub count: u32,
}

impl SymbolCounter {
    pub fn new(symbol: String, count: u32) -> Self {
        Self { symbol, count }
    }
}

impl MulAssign<u32> for SymbolCounter {
    fn mul_assign(&mut self, rhs: u32) {
        self.count *= rhs;
    }
}
