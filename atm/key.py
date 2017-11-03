from collections import namedtuple

# our KeyStruct named tuple
KeyStruct = namedtuple('KeyStruct', 'range type is_categorical')

class Key:
	TYPE_INT = "INT"
	TYPE_INT_EXP = "INT_EXP"
	TYPE_FLOAT = "FLOAT"
	TYPE_FLOAT_EXP = "FLOAT_EXP"
	TYPE_STRING = "STRING"
	TYPE_BOOL = "BOOL"

