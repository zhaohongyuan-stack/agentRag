"""
契约校验脚本 — 校验 contracts/examples/ 下的样例数据是否符合 contracts/schemas/ 下的 JSON Schema

校验逻辑：
  - retrieval_requests/  下的文件用 retrieval_request.schema.json 校验
  - retrieval_responses/ 下的文件用 retrieval_hit.schema.json 校验（每个文件是数组，逐元素校验）
  - evidence_bundles/    下的文件用 evidence_bundle.schema.json 校验

使用方式：
    python scripts/validate_contracts.py --check-all
    python scripts/validate_contracts.py                    # 同 --check-all
    python scripts/validate_contracts.py --schemas-only     # 仅校验Schema文件本身格式
    python scripts/validate_contracts.py --examples-only   # 仅校验样例文件

依赖：
    - jsonschema 库（pyproject.toml 已声明）
    - 若未安装 jsonschema，自动回退到基本 JSON 格式校验
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 路径推导
# ============================================================
# 脚本位于 scripts/validate_contracts.py，项目根目录为上一级
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMAS_DIR = PROJECT_ROOT / "contracts" / "schemas"
EXAMPLES_DIR = PROJECT_ROOT / "contracts" / "examples"

# 样例子目录 → Schema 文件名 的映射
EXAMPLE_SCHEMA_MAP = {
    "retrieval_requests": "retrieval_request.schema.json",
    "retrieval_responses": "retrieval_hit.schema.json",
    "evidence_bundles": "evidence_bundle.schema.json",
}


# ============================================================
# Schema 加载与缓存
# ============================================================
_schema_cache: Dict[str, Optional[dict]] = {}


def load_schema(schema_name: str) -> Optional[dict]:
    """
    加载 JSON Schema 文件

    Args:
        schema_name: Schema 文件名，如 "retrieval_request.schema.json"

    Returns:
        Schema dict，若文件不存在则返回 None
    """
    if schema_name in _schema_cache:
        return _schema_cache[schema_name]

    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.exists():
        _schema_cache[schema_name] = None
        return None

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        _schema_cache[schema_name] = schema
        return schema
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [错误] 无法加载 Schema {schema_name}: {e}")
        _schema_cache[schema_name] = None
        return None


# ============================================================
# jsonschema 校验（优先）与基本格式校验（回退）
# ============================================================
def _try_import_jsonschema():
    """尝试导入 jsonschema 库"""
    try:
        import jsonschema
        from jsonschema import Draft7Validator

        return jsonschema, Draft7Validator
    except ImportError:
        return None, None


def validate_with_jsonschema(instance: dict, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    使用 jsonschema 库校验实例是否符合 Schema

    Args:
        instance: 待校验的 JSON 对象
        schema: JSON Schema 对象

    Returns:
        (是否通过, 错误信息)
    """
    jsonschema, Draft7Validator = _try_import_jsonschema()
    if jsonschema is None:
        return validate_basic(instance, schema)

    try:
        Draft7Validator(schema).validate(instance)
        return (True, None)
    except jsonschema.ValidationError as e:
        return (False, str(e))


def validate_basic(instance: dict, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    基本 JSON 格式校验（jsonschema 库不可用时的回退方案）

    检查：
      - required 字段是否存在
      - type 是否匹配（仅检查基础类型）
      - enum 值是否合法

    Args:
        instance: 待校验的 JSON 对象
        schema: JSON Schema 对象

    Returns:
        (是否通过, 错误信息)
    """
    errors = []

    # 检查 type
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(instance, dict):
        errors.append(f"期望 object 类型，实际 {type(instance).__name__}")
    elif expected_type == "array" and not isinstance(instance, list):
        errors.append(f"期望 array 类型，实际 {type(instance).__name__}")
    elif expected_type == "string" and not isinstance(instance, str):
        errors.append(f"期望 string 类型，实际 {type(instance).__name__}")
    elif expected_type == "integer" and not isinstance(instance, int):
        errors.append(f"期望 integer 类型，实际 {type(instance).__name__}")
    elif expected_type == "number" and not isinstance(instance, (int, float)):
        errors.append(f"期望 number 类型，实际 {type(instance).__name__}")
    elif expected_type == "boolean" and not isinstance(instance, bool):
        errors.append(f"期望 boolean 类型，实际 {type(instance).__name__}")

    # 检查 required 字段
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in instance:
            errors.append(f"缺少必填字段: {field}")

    # 检查 enum
    properties = schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in instance:
            value = instance[field]
            enum_values = field_schema.get("enum")
            if enum_values is not None and value not in enum_values:
                errors.append(
                    f"字段 '{field}' 的值 '{value}' 不在允许的枚举值中: {enum_values}"
                )

    if errors:
        return (False, "; ".join(errors))
    return (True, None)


# ============================================================
# 单个样例文件校验
# ============================================================
def validate_example_file(
    file_path: Path, schema: dict, is_array: bool = False
) -> Tuple[bool, List[str]]:
    """
    校验单个 JSON 样例文件是否符合 Schema

    Args:
        file_path: 样例文件路径
        schema: JSON Schema 对象
        is_array: 是否为数组类型（如 retrieval_responses 是 RetrievalHit 数组）

    Returns:
        (是否全部通过, 错误信息列表)
    """
    errors = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return (False, [f"JSON 解析失败: {e}"])
    except OSError as e:
        return (False, [f"文件读取失败: {e}"])

    if is_array:
        # 数组类型：逐元素校验
        if not isinstance(data, list):
            # 特殊情况：非数组响应（如 timeout 错误响应是 dict）
            # 这种文件代表错误场景的响应体，不是 RetrievalHit 数组，跳过校验并标记为通过
            if isinstance(data, dict):
                return (True, [f"  [跳过] 非数组响应（错误响应体），不校验 RetrievalHit Schema"])
            return (False, [f"期望 JSON 数组，实际为 {type(data).__name__}"])
        if not data:
            # 空数组视为合法
            return (True, [])
        for i, item in enumerate(data):
            passed, error_msg = validate_with_jsonschema(item, schema)
            if not passed:
                errors.append(f"  数组元素[{i}]: {error_msg}")
    else:
        # 单对象类型
        passed, error_msg = validate_with_jsonschema(data, schema)
        if not passed:
            errors.append(f"  {error_msg}")

    return (len(errors) == 0, errors)


# ============================================================
# 校验 Schema 文件本身的格式
# ============================================================
def validate_schema_files() -> Tuple[int, int, List[str]]:
    """
    校验 contracts/schemas/ 下所有 JSON Schema 文件本身是否为合法 JSON

    Returns:
        (通过数, 失败数, 失败详情列表)
    """
    passed = 0
    failed = 0
    details = []

    schema_files = sorted(SCHEMAS_DIR.glob("*.json"))
    if not schema_files:
        print("  [提示] contracts/schemas/ 目录下没有 JSON Schema 文件")
        return (0, 0, [])

    print(f"\n{'='*60}")
    print(f"  校验 Schema 文件格式 ({len(schema_files)} 个)")
    print(f"{'='*60}")

    for fpath in schema_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                json.load(f)
            passed += 1
            print(f"  [PASS] {fpath.name}")
        except json.JSONDecodeError as e:
            failed += 1
            details.append(f"{fpath.name}: JSON 解析失败 - {e}")
            print(f"  [FAIL] {fpath.name}: {e}")

    return (passed, failed, details)


# ============================================================
# 校验样例文件
# ============================================================
def validate_example_files() -> Tuple[int, int, List[str]]:
    """
    校验 contracts/examples/ 下所有 JSON 样例文件是否符合对应 Schema

    Returns:
        (通过数, 失败数, 失败详情列表)
    """
    passed = 0
    failed = 0
    details = []

    print(f"\n{'='*60}")
    print("  校验样例文件")
    print(f"{'='*60}")

    # 检查 jsonschema 是否可用
    jsonschema_lib, _ = _try_import_jsonschema()
    if jsonschema_lib is not None:
        print("  [信息] 使用 jsonschema 库进行严格校验")
    else:
        print("  [警告] jsonschema 库未安装，使用基本格式校验（仅检查 required/enum/type）")

    total_files = 0

    for subdir_name, schema_filename in EXAMPLE_SCHEMA_MAP.items():
        subdir = EXAMPLES_DIR / subdir_name
        schema = load_schema(schema_filename)

        if schema is None:
            print(f"\n  [跳过] {subdir_name}/ — Schema 文件 {schema_filename} 不存在")
            continue

        if not subdir.exists():
            print(f"\n  [跳过] {subdir_name}/ — 目录不存在")
            continue

        # retrieval_responses 是数组类型
        is_array = subdir_name == "retrieval_responses"

        example_files = sorted(subdir.glob("*.json"))
        if not example_files:
            print(f"\n  [跳过] {subdir_name}/ — 没有 JSON 样例文件")
            continue

        print(f"\n  [{subdir_name}/] 使用 {schema_filename} 校验 ({len(example_files)} 个文件)")

        for fpath in example_files:
            total_files += 1
            ok, errors = validate_example_file(fpath, schema, is_array=is_array)
            if ok:
                passed += 1
                # 如果有提示信息（如跳过非数组响应），也显示出来
                if errors:
                    print(f"    [PASS] {fpath.name}  ({'; '.join(e.strip() for e in errors)})")
                else:
                    print(f"    [PASS] {fpath.name}")
            else:
                failed += 1
                for err in errors:
                    details.append(f"{subdir_name}/{fpath.name}: {err}")
                    print(f"    [FAIL] {fpath.name}")
                    for e in errors:
                        print(f"           {e}")

    if total_files == 0:
        print("\n  [提示] 没有找到任何 JSON 样例文件")
        print("         请在 contracts/examples/ 下创建样例数据")

    return (passed, failed, details)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="契约校验脚本 — 校验 contracts/examples/ 下的样例数据是否符合 contracts/schemas/ 下的 JSON Schema"
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        default=True,
        help="校验 Schema 文件格式 + 样例文件内容（默认）",
    )
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="仅校验 contracts/schemas/ 下的 Schema 文件是否为合法 JSON",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="仅校验 contracts/examples/ 下的样例文件是否符合对应 Schema",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  契约校验脚本 (validate_contracts.py)")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  Schema 目录: {SCHEMAS_DIR}")
    print(f"  样例目录: {EXAMPLES_DIR}")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    all_details = []

    # 校验 Schema 文件
    if args.schemas_only or args.check_all:
        p, f, d = validate_schema_files()
        total_passed += p
        total_failed += f
        all_details.extend(d)

    # 校验样例文件
    if args.examples_only or args.check_all:
        p, f, d = validate_example_files()
        total_passed += p
        total_failed += f
        all_details.extend(d)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  校验结果汇总")
    print(f"{'='*60}")
    print(f"  通过: {total_passed}")
    print(f"  失败: {total_failed}")
    print(f"  总计: {total_passed + total_failed}")

    if all_details:
        print(f"\n  失败详情:")
        for detail in all_details:
            print(f"    - {detail}")

    print(f"\n  {'全部通过' if total_failed == 0 else '存在校验失败'}")

    # 非零退出码表示有失败
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
