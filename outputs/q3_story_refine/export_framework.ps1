$ErrorActionPreference='Stop'
# Windows Home lacks query.exe. Supply its session-state inventory from WTS.
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Q3SessionProbe {
 [DllImport("wtsapi32.dll", SetLastError=true)]
 public static extern bool WTSQuerySessionInformationW(IntPtr server,int id,int cls,out IntPtr buffer,out int size);
 [DllImport("wtsapi32.dll")] public static extern void WTSFreeMemory(IntPtr ptr);
 public static int State(int id) { IntPtr p; int n; if(!WTSQuerySessionInformationW(IntPtr.Zero,id,8,out p,out n)) throw new Exception("WTS session query failed"); try{return Marshal.ReadInt32(p);} finally {WTSFreeMemory(p);} }
}
'@
function global:query {
 param([string]$command)
 if($command -ne 'session'){throw 'Only session inventory is supported'}
 $q3SessionId=[System.Diagnostics.Process]::GetCurrentProcess().SessionId
 $q3State=[Q3SessionProbe]::State($q3SessionId)
 $q3StateNames=@('Active','Conn','ConnectQuery','Shadow','Disc','Idle','Listen','Reset','Down','Init')
 "console $q3SessionId $($q3StateNames[$q3State])"
}
Import-Module 'C:\Users\17299\.agents\skills\.shared\office-com\scripts\office_com_common.psm1' -Force -DisableNameChecking
$q3Preflight=Get-OfficeComPreflightResult -Apps PowerPoint
$q3Preflight | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'qa/office_preflight.json') -Encoding utf8
if(-not $q3Preflight.can_use_com){throw $q3Preflight.reason}
$q3App=$null; $q3Deck=$null
try {
 $q3App=New-Object -ComObject PowerPoint.Application
 $q3Source=Join-Path $PSScriptRoot 'figure_sources/fig14_q3_framework_refined_source.pptx'
 $q3Deck=$q3App.Presentations.Open($q3Source,$true,$false,$false)
 if($q3Deck.Slides.Count -ne 1){throw 'Expected one slide'}
 $q3Deck.SaveAs((Join-Path $PSScriptRoot 'figures/q3/fig14_q3_heaf_framework.pdf'),32)
 $q3Deck.Slides.Item(1).Export((Join-Path $PSScriptRoot 'figures/q3/fig14_q3_heaf_framework.png'),'PNG',3200,1800)
} finally {
 if($null -ne $q3Deck){$q3Deck.Close();[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($q3Deck)}
 if($null -ne $q3App){$q3App.Quit();[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($q3App)}
}
