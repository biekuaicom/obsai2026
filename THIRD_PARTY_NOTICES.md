# 第三方组件与许可说明

本项目本身以 **GPL-3.0-only** 发布（见 `LICENSE`）；在开发/打包中使用以下第三方组件。**本文件不是法律意见，发布前请由法务核对实际使用方式、版本与链接方式。**

| 组件 | 版本（本机） | 许可 | 使用方式 | 风险/义务 |
|---|---|---|---|---|
| Python | 3.12.9 | PSF License | 运行时 | 保留版权声明 |
| PyQt6 / PyQt6-Qt6 | 6.9.1 / 6.9.2 | GPLv3 或 Riverbank 商业许可 | Python 绑定，动态链接 Qt | **闭源分发需购买商业许可，或按 GPLv3 开源**；建议商业版迁移 PySide6（LGPL） |
| Qt 6 | 6.9.2 | LGPLv3 / GPLv3 / 商业 | PyQt6 内含 | 动态链接/再链接义务；禁止静态链接除非合规 |
| OpenCV（opencv-python-headless） | 4.12.0.88 | Apache-2.0 | 图像处理 | 保留 NOTICE；部分 contrib 模块另有许可 |
| NumPy | 2.2.6 | BSD-3-Clause | 数值计算 | 保留版权声明 |
| PyInstaller | 6.16.0 | GPL-2.0 + bootloader 例外 | 打包工具 | 打包产物通常不受 GPL 传染，但需保留许可；以官方例外说明为准 |
| pytest | 8.4.2 | MIT | 开发测试 | 不随发布包分发 |
| Ultralytics YOLO（可选） | 未安装 | AGPL-3.0 或企业许可 | 可选检测器 | **AGPL：网络/分发使用需开源或购买企业许可**；默认包不包含 |
| Pillow（可选，未使用） | — | HPND | 图标生成备用 | 未随包分发 |
| CVAT（可选） | — | MIT（社区版） | 数据标注 | 不随包分发 |
| Label Studio（可选） | — | Apache-2.0（社区版） | 数据标注 | 不随包分发 |

## 重要提示

1. **PyQt6 是最主要的许可风险**。若软件仅内部使用、不对外分发，通常不触发 GPL 分发义务；一旦对外提供 exe/安装包，需评估 GPLv3 或购买商业许可。
2. **YOLO 适配器默认不包含 Ultralytics**。如启用并分发，需遵守 AGPL-3.0 或购买企业许可。
3. 本项目的打包产物包含 Qt/OpenCV/Python 动态库，应随包附上相应许可文本。建议在 `dist/.../licenses/` 放置各组件 LICENSE。
4. 若仅内部评估，可将本文件与 `README` 一并归档；正式商业发布前请完成许可审查。

## 建议的许可文本来源

- Qt/PyQt6: https://www.riverbankcomputing.com/commercial/license-faq
- OpenCV: https://opencv.org/license/
- NumPy: https://numpy.org/doc/stable/license.html
- PyInstaller: https://pyinstaller.org/en/stable/license.html
- Ultralytics: https://www.ultralytics.com/license
