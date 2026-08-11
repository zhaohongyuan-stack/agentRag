"""工具适配器模块"""
from .calculator import CALCULATOR_MANIFEST, calculator_handler
from .date_parser import DATE_PARSER_MANIFEST, date_parser_handler
from .version_checker import VERSION_CHECKER_MANIFEST, version_checker_handler

__all__ = [
    "CALCULATOR_MANIFEST",
    "calculator_handler",
    "VERSION_CHECKER_MANIFEST",
    "version_checker_handler",
    "DATE_PARSER_MANIFEST",
    "date_parser_handler",
]
