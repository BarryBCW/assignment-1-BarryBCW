from sys import argv
from myimplementation import test1, test2 

if __name__ == '__main__':
    argc = len( argv )

    if( argc > 1 ):
        print( "usage: python main.py" )
        exit( 1 )
       
    # run two test scrpts     
    test1()
    test2()
    