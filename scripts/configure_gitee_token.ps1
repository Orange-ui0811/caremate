param([switch]$SelfTest)

$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskEnvPath = Join-Path $taskRoot '.env'
$taskStatusPath = Join-Path $taskRoot 'data/provider-setup/gitee-token-status.json'

function Save-GiteeSettings {
    param([string]$EnvPath, [string]$Token)
    if ($Token -notmatch '^[A-Za-z0-9._~+/=-]{8,}$') {
        throw 'Token format is invalid. Paste the token only, without quotes or Bearer.'
    }
    $taskLines = [IO.File]::ReadAllLines($EnvPath)
    $taskModelsBefore = @($taskLines | Where-Object { $_ -match '^\s*[A-Z_]*MODEL\s*=' })
    $taskFoundKey = $false
    $taskFoundUrl = $false
    $taskNewLines = foreach ($taskLine in $taskLines) {
        if ($taskLine -match '^\s*CHAT_API_KEY\s*=') {
            $taskFoundKey = $true
            'CHAT_API_KEY=' + $Token
        } elseif ($taskLine -match '^\s*CHAT_BASE_URL\s*=') {
            $taskFoundUrl = $true
            'CHAT_BASE_URL=https://ai.gitee.com/v1'
        } else {
            $taskLine
        }
    }
    if (-not $taskFoundKey) { $taskNewLines += 'CHAT_API_KEY=' + $Token }
    if (-not $taskFoundUrl) { $taskNewLines += 'CHAT_BASE_URL=https://ai.gitee.com/v1' }
    $taskModelsAfter = @($taskNewLines | Where-Object { $_ -match '^\s*[A-Z_]*MODEL\s*=' })
    if (($taskModelsBefore -join "`n") -cne ($taskModelsAfter -join "`n")) {
        throw 'Model settings unexpectedly changed; nothing was saved.'
    }
    $taskTempPath = $EnvPath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        [IO.File]::WriteAllLines($taskTempPath, [string[]]$taskNewLines, [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($taskTempPath, $EnvPath, $true)
    } finally {
        if ([IO.File]::Exists($taskTempPath)) { [IO.File]::Delete($taskTempPath) }
    }
}

if ($SelfTest) {
    $taskTestDir = Join-Path $taskRoot ('data/verification/token-dialog-' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($taskTestDir) | Out-Null
    $taskTestEnv = Join-Path $taskTestDir '.env'
    $taskOriginal = @('# Test fixture', 'CHAT_MODEL=original-chat', 'SUMMARY_MODEL=',
        'EMBED_MODEL=original-embed', 'RERANK_MODEL=original-rerank',
        'CHAT_BASE_URL=https://old.invalid/v1', 'CHAT_API_KEY=old_test_token', 'OTHER_SETTING=keep')
    [IO.File]::WriteAllLines($taskTestEnv, $taskOriginal)
    Save-GiteeSettings -EnvPath $taskTestEnv -Token 'synthetic_test_token_only'
    $taskUpdated = [IO.File]::ReadAllLines($taskTestEnv)
    foreach ($taskEntry in @('CHAT_MODEL=original-chat', 'SUMMARY_MODEL=', 'EMBED_MODEL=original-embed',
                            'RERANK_MODEL=original-rerank', 'OTHER_SETTING=keep',
                            'CHAT_BASE_URL=https://ai.gitee.com/v1', 'CHAT_API_KEY=synthetic_test_token_only')) {
        if ($taskEntry -notin $taskUpdated) { throw 'Settings preservation test failed.' }
    }
    Write-Output 'PASS: provider/key updated; all model and unrelated settings preserved. Production .env untouched.'
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$taskForm = [Windows.Forms.Form]::new()
$taskForm.Text = 'Gitee AI 令牌配置'
$taskForm.Size = [Drawing.Size]::new(610, 315)
$taskForm.StartPosition = 'CenterScreen'
$taskForm.FormBorderStyle = 'FixedDialog'
$taskForm.MaximizeBox = $false
$taskForm.MinimizeBox = $false
$taskForm.TopMost = $true
$taskForm.Font = [Drawing.Font]::new('Microsoft YaHei UI', 10)

$taskLabel = [Windows.Forms.Label]::new()
$taskLabel.Text = "请先复制 Gitee AI 访问令牌，再点击下方粘贴按钮。`r`n令牌仅保存到本机 .env，所有模型配置保持不变。"
$taskLabel.Location = [Drawing.Point]::new(20, 20)
$taskLabel.Size = [Drawing.Size]::new(560, 55)
$taskForm.Controls.Add($taskLabel)

$taskInput = [Windows.Forms.TextBox]::new()
$taskInput.Location = [Drawing.Point]::new(20, 85)
$taskInput.Size = [Drawing.Size]::new(550, 30)
$taskInput.UseSystemPasswordChar = $true
$taskInput.ShortcutsEnabled = $true
$taskForm.Controls.Add($taskInput)

$taskPaste = [Windows.Forms.Button]::new()
$taskPaste.Text = '从剪贴板粘贴'
$taskPaste.Location = [Drawing.Point]::new(20, 125)
$taskPaste.Size = [Drawing.Size]::new(170, 35)
$taskForm.Controls.Add($taskPaste)

$taskMessage = [Windows.Forms.Label]::new()
$taskMessage.Location = [Drawing.Point]::new(20, 170)
$taskMessage.Size = [Drawing.Size]::new(550, 35)
$taskMessage.ForeColor = [Drawing.Color]::Firebrick
$taskForm.Controls.Add($taskMessage)

function Paste-GiteeToken {
    try {
        if ([Windows.Forms.Clipboard]::ContainsText()) {
            $taskInput.Text = [Windows.Forms.Clipboard]::GetText().Trim()
            $taskInput.SelectionStart = $taskInput.TextLength
            $taskMessage.ForeColor = [Drawing.Color]::ForestGreen
            $taskMessage.Text = '已粘贴，点击“保存配置”即可。'
        } else {
            $taskMessage.ForeColor = [Drawing.Color]::Firebrick
            $taskMessage.Text = '剪贴板中没有文本，请先复制访问令牌。'
        }
    } catch {
        $taskMessage.ForeColor = [Drawing.Color]::Firebrick
        $taskMessage.Text = '剪贴板暂不可用，请重新复制后再次点击。'
    }
    $taskInput.Focus()
}
$taskPaste.Add_Click({ Paste-GiteeToken })
$taskInput.Add_KeyDown({
    param($sender, $eventArgs)
    if ($eventArgs.Control -and $eventArgs.KeyCode -eq [Windows.Forms.Keys]::V) {
        Paste-GiteeToken
        $eventArgs.Handled = $true
        $eventArgs.SuppressKeyPress = $true
    }
})
$taskMenu = [Windows.Forms.ContextMenuStrip]::new()
$taskPasteMenu = $taskMenu.Items.Add('粘贴令牌')
$taskPasteMenu.Add_Click({ Paste-GiteeToken })
$taskInput.ContextMenuStrip = $taskMenu

$taskSave = [Windows.Forms.Button]::new()
$taskSave.Text = '保存配置'
$taskSave.Location = [Drawing.Point]::new(320, 215)
$taskSave.Size = [Drawing.Size]::new(125, 35)
$taskForm.Controls.Add($taskSave)
$taskCancel = [Windows.Forms.Button]::new()
$taskCancel.Text = '取消'
$taskCancel.Location = [Drawing.Point]::new(455, 215)
$taskCancel.Size = [Drawing.Size]::new(115, 35)
$taskForm.Controls.Add($taskCancel)
$taskForm.AcceptButton = $taskSave
$taskForm.CancelButton = $taskCancel

$taskSave.Add_Click({
    try {
        Save-GiteeSettings -EnvPath $taskEnvPath -Token $taskInput.Text.Trim()
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskStatusPath)) | Out-Null
        $taskStatus = @{ configured = $true; models_unchanged = $true;
            base_url = 'https://ai.gitee.com/v1'; saved_at_utc = [DateTime]::UtcNow.ToString('o') }
        [IO.File]::WriteAllText($taskStatusPath, ($taskStatus | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
        $taskInput.Clear()
        $taskForm.DialogResult = [Windows.Forms.DialogResult]::OK
        $taskForm.Close()
    } catch {
        $taskMessage.ForeColor = [Drawing.Color]::Firebrick
        $taskMessage.Text = '保存失败：请仅粘贴令牌，不要包含 Bearer、空格或引号。'
    }
})
$taskCancel.Add_Click({ $taskInput.Clear(); $taskForm.Close() })
$taskForm.Add_Shown({ $taskForm.Activate(); $taskInput.Focus() })
Write-Output 'Local password dialog opened; waiting for the user. Token will not be printed.'
$taskResult = $taskForm.ShowDialog()
$taskInput.Clear()
$taskForm.Dispose()
if ($taskResult -eq [Windows.Forms.DialogResult]::OK) {
    Write-Output 'SAVED: Gitee chat endpoint and token configured; all model settings unchanged.'
} else {
    Write-Output 'CANCELLED: no token submitted.'
}
