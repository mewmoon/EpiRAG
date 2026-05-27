from pathlib import Path
from typing import Union


class PathUtils:
    # 获取当前文件的父目录的父目录 (假设此文件在项目根目录的子文件夹中)
    # 调整 .parent 的数量即可控制向上的层级：1为父，2为祖父，以此类推
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    @staticmethod
    def to_absolute(relative_path: Union[str, Path]) -> str:
        """将路径转换为绝对路径"""
        # 如果传入的是相对路径，则基于 PROJECT_ROOT 进行解析
        return str((PathUtils.PROJECT_ROOT / relative_path).resolve())

    @staticmethod
    def to_relative(
        absolute_path: Union[str, Path], start_path: Union[str, Path] = "."
    ) -> str:
        """将绝对路径转为相对于 start_path 的路径"""
        return str(
            Path(absolute_path).resolve().relative_to(Path(start_path).resolve())
        )
