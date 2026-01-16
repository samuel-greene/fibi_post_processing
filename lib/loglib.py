from datetime import datetime
from enum import Enum

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Error:
	class ErrorBase:
		exit_process_on_throw = True
		default_log_str = ""

		# default log string property can be overridden by child classes
		# if child class has 'default_log_str' attribute, use that as default log str
		# and pass args to be input in the format of the log string
		# if not, then use the throw_str passed to throw() method
		@classmethod
		def throw(cls,throw_str = "", *args: tuple):
			timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
			error = f"[Error.{cls.__name__}]" if cls.exit_process_on_throw else f"[Error.{cls.__name__} : Execution Aborted]"

			if cls.default_log_str != "":
				args = (throw_str,) + args
				print(f"{timestamp}{bcolors.FAIL}{error} : {cls.default_log_str.format(*args)}{bcolors.ENDC}")
				Debug.debug("Found a default log string for this error type.")
			else:
				print(f"{timestamp}{bcolors.FAIL}{error} : {throw_str}{bcolors.ENDC}")
				Debug.debug("No default log string for this error type.")
			
			if cls.exit_process_on_throw:
				exit(0)
		
		@classmethod
		def set_log_str(cls, log_str: str):
			cls.default_log_str = log_str
		@classmethod
		def set_exit_on_throw(cls, exit_on_throw: bool):
			cls.exit_process_on_throw = exit_on_throw

	# Classic Errors
	class FileNotFound(ErrorBase):
		exit_process_on_throw = True
		log_str = "File '{}' not found"
	class InvalidFileFormat(ErrorBase):
		exit_process_on_throw = True
		log_str = "File '{}' has an invalid format" 
	class InvalidPath(ErrorBase):
		exit_process_on_throw = True
		log_str = "Path '{}' is invalid"
	class PermissionDenied(ErrorBase):
		exit_process_on_throw = True
		log_str = "Permission denied for path '{}'"

class Debug:
	class LogLevel(Enum):
		QUIET = 0
		CONSOLE = 1
		FILE = 2
		VERBOSE = 3

	_mode = LogLevel.VERBOSE
 
	@classmethod
	def base_log(cls, type: str, log_str: str):
		timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
		log_color = bcolors.OKCYAN
		if type == "Warning":
			log_color = bcolors.WARNING
		elif type == "Error":
			log_color = bcolors.FAIL
   
		match cls._mode:
			case cls.LogLevel.CONSOLE:
				print(f"{timestamp}{log_color} [{type}] : {log_str}{bcolors.ENDC}")
			case cls.LogLevel.FILE:
				with open("log.txt", "a") as log_file:
					log_file.write(f"{timestamp} [{type}] : {log_str}\n")
			case cls.LogLevel.VERBOSE:
				print(f"{timestamp}{log_color} [{type}] : {log_str}{bcolors.ENDC}")
				with open("log.txt", "a") as log_file:
					log_file.write(f"{timestamp} [{type}] : {log_str}\n")
			case cls.LogLevel.QUIET:
				pass

	@classmethod
	def log(cls, log_str: str):
		cls.base_log(type="INFO", log_str=log_str)
	@classmethod
	def warning(cls, log_str: str):
		cls.base_log(type="Warning", log_str=log_str)
	@classmethod
	def error(cls, log_str: str):
		cls.base_log(type="Error", log_str=log_str)

	@classmethod
	def debug(cls, log_str: str):
		print(f"{bcolors.OKBLUE} [Live Debug] : {log_str}{bcolors.ENDC}")
	
	@classmethod
	def set_mode(cls, mode: LogLevel):
		cls._mode = mode


# Clear log file on import
with open("log.txt", "w") as log_file:
	log_file.write("")