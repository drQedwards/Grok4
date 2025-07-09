/* eslint-disable no-console */

interface LogLevel {
  info: (...args: any[]) => void;
  error: (...args: any[]) => void;
  warn: (...args: any[]) => void;
  debug: (...args: any[]) => void;
}

class Logger implements LogLevel {
  private formatMessage(level: string, message: any): string {
    const timestamp = new Date().toISOString();
    return `[${timestamp}] [${level}] ${typeof message === 'object' ? JSON.stringify(message, null, 2) : message}`;
  }

  info(...args: any[]): void {
    console.log(this.formatMessage('INFO', args.join(' ')));
  }

  error(...args: any[]): void {
    console.error(this.formatMessage('ERROR', args.join(' ')));
  }

  warn(...args: any[]): void {
    console.warn(this.formatMessage('WARN', args.join(' ')));
  }

  debug(...args: any[]): void {
    if (process.env.NODE_ENV === 'development') {
      console.log(this.formatMessage('DEBUG', args.join(' ')));
    }
  }
}

export const logger = new Logger();