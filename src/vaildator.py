# coding: utf-8

# Author: Mingzhe Du (mingzhe@nus.edu.sg)
# Date: 2024/09/01

class Vaildator:
    def verify(self, code) -> bool:
        raise NotImplementedError("Don't call Base Vaildator")
    
class CodeExecutable(Vaildator):
    def verify(self, code, cases) -> bool:
