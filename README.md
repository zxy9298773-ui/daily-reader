# daily-reader

外刊每日推送小工具：RSS 抓取 → DeepSeek 翻译 + 词汇提取 → 生成 HTML 邮件推送。

## 运行

```
python main.py           # 试运行，输出到控制台
python main.py --send    # 真正发送邮件
```

## 配置

复制 `.env.example` 为 `.env`，填入：

- `DEEPSEEK_API_KEY` — DeepSeek API 密钥（翻译/词汇提取）
- `RESEND_API_KEY` — Resend 发信密钥
- `TO_EMAIL` — 收件邮箱

## 说明

- 定时推送（GitHub Actions）已于 **2026-10-09 停用**，需要时在仓库 Actions 页面重新启用即可。
- 无意义的每日 README 自动提交已清理并删除对应 workflow。
