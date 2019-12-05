import fnmatch
import os
import time
import atexit
from SCons.Defaults import *

release = True

if(release):
	optimization = ['-O3', '-DNDEBUG']
	debug = '-g0'
	lto = "1"
	closure = "0"
	assertions = "0"
	demangle = "0"
else:
	optimization = ['-O0']
	debug = '-g3'
	lto = "0"
	closure = "0"
	assertions = "2"
	demangle = "1"


def main():
	env = Environment(ENV = os.environ, tools = ['gcc', 'g++', 'gnulink', 'ar', 'gas'])
		
	env.Replace(CC     = "emcc"    )
	env.Replace(CXX    = "em++"    )
	env.Replace(LINK   = "emcc"    )
	
	env.Replace(AR     = "emar"    )
	env.Replace(RANLIB = "emranlib")
	
	env.Replace(LIBLINKPREFIX = "")
	env.Replace(LIBPREFIX = "")
	env.Replace(LIBLINKSUFFIX = ".bc")
	env.Replace(LIBSUFFIX = ".bc")
	env.Replace(OBJSUFFIX  = ".o")
	env.Replace(PROGSUFFIX = ".html")
	
	env.Append( CPPFLAGS=optimization)
	env.Append( LINKFLAGS=[
		optimization,
		debug,
		"-lGL",
		"-s", "ASSERTIONS=" + assertions,
		"-s", "DEMANGLE_SUPPORT=" + demangle,
        #"-s", "ALLOW_MEMORY_GROWTH=0",
		"-s", "TOTAL_MEMORY=1024MB",
        "-s", "EXTRA_EXPORTED_RUNTIME_METHODS=[\"ccall\", \"cwrap\"]",
		"--llvm-lto", lto,
		"--closure", closure,
		"-s", "NO_EXIT_RUNTIME=1",
		"-s", "DISABLE_EXCEPTION_CATCHING=1",
		"--bind",
		"--preload-file", "StyleGAN.bin"]
	)

	timeStart = time.time()
	atexit.register(PrintInformationOnBuildIsFinished, timeStart)
	
	Includes = [
		"tensor4/examples/common",
		"tensor4/include",
	]
	
	files = ["main.cpp", "StyleGAN.cpp"]
	
	program = env.Program('stylegan', files, LIBS=[], CPPFLAGS=optimization + ['-std=c++14',  debug], LIBPATH='.', CPPPATH = Includes)
	
	
def PrintInformationOnBuildIsFinished(startTimeInSeconds):
	""" Launched when scons is finished """
	failures = GetBuildFailures()
	for failure in failures:
		print("Target [%s] failed: %s" % (failure.node, failure.errstr))
	timeDelta = time.gmtime(time.time() - startTimeInSeconds)
	print(time.strftime("Build time: %M minutes %S seconds", timeDelta))
	
def GlobR(path, filter) : 
	matches = []
	for root, dirnames, filenames in os.walk(path):
  		for filename in fnmatch.filter(filenames, filter):
   			matches.append(os.path.join(root, filename)) 
	return matches

if __name__ == "SCons.Script":
	main()
