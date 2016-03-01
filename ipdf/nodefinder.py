import ply.lex as lex
import ply.yacc as yacc

class nodeFinder(object):

    def __init__(self):


        self.tokens = ("SLASH", "DOTDOT", "NUMBER", "DOT", "LPAREN", "RPAREN")
        self.t_SLASH = r'/'
        self.t_DOTDOT = r'\.\.'
        self.t_DOT = r'\.'
        self.t_LPAREN = r'\('
        self.t_RPAREN = r'\)'

        self.lexer = lex.lex(module=self)
        self.yaccer = yacc.yacc(module=self)

        self.root = None
        self.node = None

    def t_NUMBER(self,t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_error(self,t):
        print("Illegal term '%s'" % t.value[0])

    def p_term_slash_path(self,p):
        'path : path SLASH term'
        self.node = p[1]
        p[0] = p[3]

    def p_path_term(self,p):
        'path : term'
        p[0] = p[1]

    def p_path_slash(self,p):
        'path : SLASH'
        p[0] = self.root

    def p_addr_dot(self,p):
        'addr : addr DOT term'
        self.node = p[1]
        p[0] = p[3]

    def p_addr_term(self, p):
        'addr : term'
        p[0] = p[1]

    def p_term_num(self,p):
        'term : NUMBER'
        self.node = self.node.children[p[1]]
        p[0] = self.node

    def p_term_dotdot(self,p):
        'term : DOTDOT'
        self.node = self.node.parent
        p[0] = self.node

    def p_term_addr(self, p):
        'term : LPAREN addr RPAREN'
        p[0] = p[2]

    def __call__(self, path, root, cwn):

        self.root = root
        self.node = cwn
        self.args = {}

        return self.yaccer.parse(path)
