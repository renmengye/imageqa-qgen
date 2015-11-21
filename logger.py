"""
A python logger

Usage:
    # Set logger verbose level.
    import os
    os.environ['VERBOSE'] = 1

    import logger
    log = logger.get('../logs/sample_log')
    
    log.info('Hello world!')
    log.info('Hello again!', verbose=2)
    log.warning('Something might be wrong.')
    log.error('Something is wrong.')
    log.fatal('Failed.')
"""

from __future__ import print_function
import datetime
import inspect
import os
import sys

terminal = {
    'normal': '\033[0m',
    'bright': '\033[1m',
    'invert': '\033[7m',
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'default': '\033[39m'
}

log = None


def get(default_fname=None):
    """
    Returns a logger instance, with optional log file output.
    """
    global log
    if log is not None:
        return log
    fname = os.environ.get('LOGTO', None)
    if fname is None:
        fname = default_fname
    log = Logger(fname)
    return log


class Logger(object):

    def __init__(self, filename=None):
        """
        Constructs a logger with optional log file output.

        Args:
            filename: optional log file output. If None, nothing will be 
            written to file
        """
        now = datetime.datetime.now()
        self.verbose_thresh = os.environ.get('VERBOSE', 0)
        if filename is not None:
            self.filename = \
                '{}-{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}.log'.format(
                    filename,
                    now.year, now.month, now.day,
                    now.hour, now.minute, now.second)
            dirname = os.path.dirname(self.filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            open(self.filename, 'w').close()
            self.info('Log written to {}'.format(
                os.path.abspath(self.filename)))
        else:
            self.filename = None
        pass

    @staticmethod
    def get_time_str(t=None):
        """
        Returns a formatted time string.

        Args:
            t: datetime, default now.
        """
        if t is None:
            t = datetime.datetime.now()

        timestr = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
            t.year, t.month, t.day, t.hour, t.minute, t.second)

        return timestr

    def log(self, message, typ='info', verbose=0):
        """
        Writes a message.

        Args:
            message: string, message content.
            typ: string, type of the message. info, warning, error, or fatal.
            verbose: number, verbose level of the message. If lower than the 
            environment variable, then the message will be logged to standard 
            output and log output file (if set).
        """
        if typ == 'info':
            typstr_print = '{0}INFO:{1}'.format(
                terminal['green'], terminal['default'])
            typstr_log = 'INFO:'
        elif typ == 'warning':
            typstr_print = '{0}WARNING:{1}'.format(
                terminal['yellow'], terminal['default'])
            typstr_log = 'WARNING'
        elif typ == 'error':
            typstr_print = '{0}ERROR:{1}'.format(
                terminal['red'], terminal['default'])
            typstr_log = 'ERROR'
        elif typ == 'fatal':
            typstr_print = '{0}FATAL:{1}'.format(
                terminal['red'], terminal['default'])
            typstr_log = 'FATAL'
        else:
            raise Exception('Unknown log type: {0}'.format(typ))
        timestr = self.get_time_str()
        for (frame, filename, line_number, function_name, lines, index) in \
                inspect.getouterframes(inspect.currentframe()):
            if not filename.endswith('logger.py'):
                break
        cwd = os.getcwd()
        if filename.startswith(cwd):
            filename = filename[len(cwd):]
        filename = filename.lstrip('/')
        callerstr = '{0}:{1}'.format(filename, line_number)
        printstr = '{0} {1} {2} {3}'.format(
            typstr_print, timestr, callerstr, message)
        logstr = '{0} {1} {2} {3}'.format(
            typstr_log, timestr, callerstr, message)
        if self.verbose_thresh <= verbose:
            print(printstr)
        if self.filename is not None:
            with open(self.filename, 'a') as f:
                f.write(logstr)
                f.write('\n')
        pass

    def info(self, message, verbose=0):
        """
        Writes an info message.

        Args:
            message: string, message content.
            verbose: number, verbose level.
        """
        self.log(message, typ='info', verbose=verbose)
        pass

    def warning(self, message, verbose=0):
        """
        Writes a warning message.

        Args:
            message: string, message content.
            verbose: number, verbose level.
        """
        self.log(message, typ='warning', verbose=verbose)
        pass

    def error(self, message, verbose=0):
        """
        Writes an info message.

        Args:
            message: string, message content.
            verbose: number, verbose level.
        """
        self.log(message, typ='error', verbose=verbose)
        pass

    def fatal(self, message, verbose=0):
        """
        Writes a fatal message, and exits the program.

        Args:
            message: string, message content.
            verbose: number, verbose level.
        """
        self.log(message, typ='fatal', verbose=verbose)
        sys.exit(0)
        pass

    def log_args(self, verbose=0):
        self.info('Command: {}'.format(' '.join(sys.argv)))
