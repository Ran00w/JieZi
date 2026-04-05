# JieZi 从 0 开始维护手册（Git/GitHub）

适用仓库：`https://github.com/Ran00w/JieZi.git`  
当前状态（2026-04-05）：仅 `main` 分支、1 个初始提交、仅 `LICENSE` 文件。

## 1. 先建立正确心智（5 分钟）

- `工作区`：你电脑上的文件。
- `暂存区`：`git add` 后，准备提交的内容。
- `本地仓库`：`git commit` 后形成历史。
- `远程仓库`：GitHub 上的仓库（`origin`）。

你每天的基本循环就是：
`拉最新 -> 新建分支 -> 改代码 -> 提交 -> 推送 -> 提 PR -> 合并 -> 同步 main`

## 2. 一次性初始化（只做一次）

在终端执行（把姓名和邮箱改成你自己的）：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
```

检查是否成功：

```bash
git config --global --list
```

## 3. 克隆与检查（你的仓库已经完成）

```bash
git clone https://github.com/Ran00w/JieZi.git
cd JieZi
git branch -a
git log --oneline --decorate -n 5
git remote -v
```

## 4. 日常开发标准流程（最重要）

### 4.1 每天开始先同步

```bash
git checkout main
git pull origin main
```

### 4.2 开新分支做事情（不要直接改 main）

```bash
git checkout -b chore/bootstrap-readme
```

分支命名建议：
- `feat/xxx`：新功能
- `fix/xxx`：修 bug
- `docs/xxx`：文档
- `chore/xxx`：维护工作

### 4.3 改完后提交

```bash
git status
git add .
git commit -m "docs: add initial README and maintenance guide"
```

提交信息建议格式：
`type: short summary`，常用 `feat fix docs chore refactor test`

### 4.4 推送并发起 PR

```bash
git push -u origin chore/bootstrap-readme
```

然后去 GitHub 页面创建 PR（base: `main`, compare: 你的分支）。

### 4.5 合并后清理

```bash
git checkout main
git pull origin main
git branch -d chore/bootstrap-readme
git push origin --delete chore/bootstrap-readme
```

## 5. 版本发布（推荐）

当 main 达到一个可发布状态时：

```bash
git checkout main
git pull origin main
git tag -a v0.1.0 -m "first usable version"
git push origin v0.1.0
```

版本号建议：
- `v0.x.x`：早期迭代
- `v1.0.0`：第一个稳定版本
- `MAJOR.MINOR.PATCH`：语义化版本

## 6. 回滚与救火（必须会）

### 撤销工作区改动（未 add）

```bash
git restore <file>
```

### 撤销暂存区（已 add 未 commit）

```bash
git restore --staged <file>
```

### 用新提交回滚历史（已 push，团队协作推荐）

```bash
git revert <commit_id>
git push
```

避免直接 `reset --hard` 改公共历史。

## 7. 公开仓库维护建议

- 给 `main` 开启保护：禁止直接 push、必须 PR。
- 至少 1 个 PR 模板（说明改动、测试、风险）。
- 每次改动至少写清楚：做了什么、为什么、怎么验证。
- 小步提交，避免“大杂烩”提交。

## 8. 你现在可以立刻做的第一轮实操

1. 新建分支：`git checkout -b docs/init-readme`
2. 新增 `README.md`（写项目简介、怎么运行、目录说明）。
3. `git add README.md MAINTAINING_FROM_ZERO_ZH.md`
4. `git commit -m "docs: add initial README and maintenance guide"`
5. `git push -u origin docs/init-readme`
6. 在 GitHub 发 PR，合并后删分支。

## 9. 常用排错命令

```bash
git status
git branch -vv
git log --oneline --decorate --graph -n 20
git diff
git diff --staged
```

如果你愿意，下一步我可以直接带你做第 1 次完整实操：  
从创建 `README.md` 分支开始，到推送、PR、合并、打 `v0.1.0` 标签，整套跑完一次。
