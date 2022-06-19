use super::enums::*;

#[derive(Default)]
pub struct ElementData {
    pub symbol: String,
    pub name: String,
    pub atomic_weight: f32
}

impl ElementData {
    pub fn new() -> Self {
        let mut e = Self::default();
        e.initialize();
        e
    }

    fn initialize(&mut self) {}
}

pub struct AllElementData {
    pub data: HashMap<Symbol, ElementData>
}

impl AllElementData {
    pub fn new() -> Self {
    }
}
