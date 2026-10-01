@echo off
:: 1. 设置网络代理（请确保端口号 10809 与你的代理软件一致）
set http_proxy=http://127.0.0.1:10809
set https_proxy=http://127.0.0.1:10809

:: 2. 打印提示信息，方便确认状态
echo [INFO] Proxy has been set to 127.0.0.1:10809
echo [INFO] Starting Gemini CLI with thesis context...

:: 3. 启动 Gemini CLI 并注入核心目录
gemini 
:: 4. 如果程序退出，保持窗口不关闭（方便查看报错）
pause