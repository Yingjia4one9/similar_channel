Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = oWS.SpecialFolders("Desktop") & "\YouTube相似频道.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\start.bat"
oLink.WorkingDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
oLink.Description = "Start YouTube Similar Channel Service"
oLink.IconLocation = "C:\Windows\System32\shell32.dll,21"
oLink.Save
WScript.Echo "快捷方式创建成功！桌面位置: " & sLinkFile

