import fnmatch
import os
import time
import atexit
from SCons.Defaults import *

release = True

if(release):
	optimization = ['-O3', '-DNDEBUG', '-fno-rtti', '-fno-exceptions', '-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0']
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
        "-s", "ALLOW_MEMORY_GROWTH=0",
		"-s", "TOTAL_MEMORY=1023MB",
        "-s", "EXTRA_EXPORTED_RUNTIME_METHODS=[\"ccall\", \"cwrap\"]",
		"--llvm-lto", lto,
		"--closure", closure,
		"-s", "NO_EXIT_RUNTIME=1",
		"-s", "DISABLE_EXCEPTION_CATCHING=1",
		"--bind",
		"--preload-file", "StyleGAN.ct4"]
	)

	timeStart = time.time()
	atexit.register(PrintInformationOnBuildIsFinished, timeStart)
	
	Includes = [
		"tensor4/examples/common",
		"tensor4/include",
		"zfp/include",
	]

	files = ["main.cpp", "StyleGAN.cpp"]
	zfp = Glob("zfp/src", "*.c")

	zfpl = env.Library('zfplib', zfp, LIBS=[], CPPFLAGS=optimization + [debug], LIBPATH='.', CPPPATH = Includes)
	program = env.Program('stylegan', files, LIBS=[zfpl], CPPFLAGS=optimization + ['-std=c++14',  debug], LIBPATH='.', CPPPATH = Includes)
	
	
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

def Glob(path, filter) :
	matches = []
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	for filename in fnmatch.filter(onlyfiles, filter):
		matches.append(os.path.join(path, filename))
	return matches

if __name__ == "SCons.Script":
	main()
