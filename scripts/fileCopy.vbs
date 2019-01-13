Option Explicit

' The source path for the copy operation.
Const strSourceFolder = "H:"

' The target path for the copy operation.
Const strTargetFolder = "C:\Temp\Target\"




' The list of files to copy. Should be a text file with one file on each row. No paths - just file name.
Const strFileList = "C:\filelist.txt"

' Should files be overwriten if they already exist? TRUE or FALSE.
Const blnOverwrite = FALSE





Dim objFSO
Set objFSO = CreateObject("Scripting.FileSystemObject")

Const ForReading = 1
Dim objFileList
Set objFileList = objFSO.OpenTextFile(strFileList, ForReading, False)

Dim strFileToCopy, strSourceFilePath, strTargetFilePath
On Error Resume Next
Do Until objFileList.AtEndOfStream
    ' Read next line from file list and build filepaths
    strFileToCopy = objFileList.Readline
    strSourceFilePath = objFSO.BuildPath(strSourceFolder, strFileToCopy)
    strTargetFilePath = objFSO.BuildPath(strTargetFolder, strFileToCopy)

    ' Copy file to specified target folder.
    Err.Clear
    objFSO.CopyFile strSourceFilePath, strTargetFilePath, blnOverwrite
    If Err.Number = 0 Then
        ' File copied successfully

    Else
        ' Error copying file
        Wscript.Echo "Error " & Err.Number & " (" & Err.Description & "). Copying " & strFileToCopy
    End If
Loop