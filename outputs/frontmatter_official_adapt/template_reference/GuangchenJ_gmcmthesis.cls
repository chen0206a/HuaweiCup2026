%
% This is file `gmcmthesis.cls',
% created by latexstudio.net
% update by andy123t
%
% 版权与许可：
%   本文件派生自 GMCMthesis 模板（latexstudio.net、andy123t、
%   OsbertWang/GMCMthesis），保留其全部原作者与维护者署名。
%   本文件按 LaTeX Project Public License 1.3c 或更高版本发布，
%   许可全文见仓库根目录 LICENSE，来源与第三方组件说明见 README.md。
%
% v2.5 (2026/09/22) 重构说明：
%   * 附录编号自动处理：\begin{appendices} 后小节/图表/公式编号自动变为
%     附录A、附录A.1……，无需再手动 \renewcommand\thesubsection
%   * 新增普通样式的「定义/定理/假设/问题」等环境（加粗标签 + 宋体正文，
%     符合官方格式规范）、灰色 keybox 提示框、note/proof/solution 环境
%   * 新增中文 \autoref 名称与 \figref/\tabref/\eqnref 引用简写、
%     \circled 带圈序号、siunitx 单位命令；algorithm 算法环境已内置
%   * 修复：重复加载 listings、Python 代码环境缺逗号、ulem 覆盖 \emph、
%     hyperref 重复选项警告等
%   * 格式对齐官方规范：题目三号黑体、一级标题四号黑体居中、
%     其余汉字小四宋体、单倍行距、无页眉
%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{gmcmthesis}[2026/09/22 v2.5 update by andy123t]
%% Inital Code
\newif\if@gmcm@bwprint\@gmcm@bwprintfalse
\newif\if@gmcm@preface\@gmcm@prefacetrue
% tokens definition
%%%% 承诺书
\newcommand\gmcm@tokens@keywords{}
\newcommand*\gmcm@tokens@tihao{}
\newcommand*\gmcm@tokens@baominghao{}
\newcommand*\gmcm@tokens@schoolname{}
\newcommand*\gmcm@tokens@membera{}
\newcommand*\gmcm@tokens@memberb{}
\newcommand*\gmcm@tokens@memberc{}
\newcommand*\gmcm@tokens@supervisor{}
\newcommand*\gmcm@tokens@yearinput{}
\newcommand*\gmcm@tokens@monthinput{}
\newcommand*\gmcm@tokens@dayinput{}
%%%% 编号专用页
% 没有
%% Declaration of Options
\DeclareOption{colorprint}{\@gmcm@bwprintfalse}
\DeclareOption{bwprint}{\@gmcm@bwprinttrue}
\DeclareOption{withoutpreface}{\@gmcm@prefacefalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
%% Executing of Options
\ExecuteOptions{colorprint}
\ProcessOptions\relax
%% Loading Class
\LoadClass[a4paper,zihao=-4]{ctexart}
%% Loading Package
\RequirePackage{iftex}
\RequireXeTeX
\ifxetex\else
\ClassError{gmcmthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi
% 页面布局
\RequirePackage{geometry}
% 数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
% 设置颜色
\RequirePackage{xcolor}
% 插入图片
\RequirePackage{graphicx}
% 表格
\RequirePackage{tabularx,array}
% 调整单元格格式
\RequirePackage{makecell}
%% 长表格
\RequirePackage{longtable}
%% booktabs 提供了\toprule 等命令.
\RequirePackage{booktabs}
%% multirow 支持在表格中跨行
\RequirePackage{multirow}
%% 调整间隔, 让表格更好看些
\RequirePackage{bigstrut}
%% 在跨行表格中输入定界符
\RequirePackage{bigdelim}
% 保护脆弱命令
\RequirePackage{cprotect}
% 代码环境
\RequirePackage{listings}
% 首行缩进
\RequirePackage{indentfirst}
% 浮动体标题与子图
\RequirePackage{caption}
\RequirePackage{subcaption}
% 定制列表环境
\RequirePackage{enumitem}
% 下划线（normalem 选项保证不改变 \emph 的语义）
\RequirePackage[normalem]{ulem}
% 尺寸计算
\RequirePackage{calc}
\RequirePackage{url}
% 参考文献
\RequirePackage[numbers]{natbib}
\setlength{\bibsep}{0pt plus 0.3ex}
% 单位与数字（数模论文常用，如 \SI{70}{cm}）
\RequirePackage{siunitx}
% 彩色盒子（定理类环境与提示框的基础）
\RequirePackage[most]{tcolorbox}
% 算法
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\floatname{algorithm}{算法}
\renewcommand{\algorithmicrequire}{\textbf{输入:}}
\renewcommand{\algorithmicensure}{\textbf{输出:}}
% 超链接
\RequirePackage{hyperref}
\hypersetup{%
  colorlinks=true,
  allcolors=black,
  breaklinks=true,
  bookmarksnumbered=true,
  bookmarksopen=true,
  bookmarksopenlevel=0,
}
% URL 断行设置
\def\UrlAlphabet{%
      \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j%
      \do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t%
      \do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D%
      \do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N%
      \do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X%
      \do\Y\do\Z}
\def\UrlDigits{\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9\do\0}
\g@addto@macro{\UrlBreaks}{\UrlOrds}
\g@addto@macro{\UrlBreaks}{\UrlAlphabet}
\g@addto@macro{\UrlBreaks}{\UrlDigits}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Document Layout
% 页面布局
\geometry{top=30.0mm,bottom=25.0mm,left=22.5mm,right=22.5mm,headsep=8mm}
\renewcommand*{\baselinestretch}{1.0}
% 每行缩进两个汉字
\setlength\parindent{2em}
% 设置字体
% 代码来自https://github.com/OsbertWang/GMCMthesis 这个是OsbertWang，修正支持让各个系统自适应支持近似字体。
\newif\ifmcm@mac
\newif\ifmcm@windows
\newif\ifmcm@fandol


\@ifclasslater{ctexart}{2015/05/06}{
  %\newcommand\mcm@strifeq{\csname str_if_eq_x:nnTF\endcsname}
  %修订错误来自：https://github.com/ustctug/ustcthesis/issues/241
  \newcommand\mcm@strifeq[4]{%
  \edef\ustc@tmp@a{#1}%
  \edef\ustc@tmp@b{#2}%
  \ifx\ustc@tmp@a\ustc@tmp@b
    #3%
  \else
    #4%
  \fi
}
  \newcommand\mcm@fontset{\csname g__ctex_fontset_tl\endcsname}
  \mcm@strifeq{\mcm@fontset}{mac}{\mcm@mactrue}{}
  \mcm@strifeq{\mcm@fontset}{windows}{\mcm@windowstrue}{}
  \mcm@strifeq{\mcm@fontset}{fandol}{\mcm@fandoltrue}{}
}{
  \mcm@windowstrue
}

\ifmcm@fandol
  \setmainfont[
    Extension      = .otf,
    UprightFont    = *-regular,
    BoldFont       = *-bold,
    ItalicFont     = *-italic,
    BoldItalicFont = *-bolditalic,
  ]{texgyretermes}
  \setsansfont[
    Extension      = .otf,
    UprightFont    = *-regular,
    BoldFont       = *-bold,
    ItalicFont     = *-italic,
    BoldItalicFont = *-bolditalic,
  ]{texgyreheros}
  \setmonofont[
    Extension      = .otf,
    UprightFont    = *-regular,
    BoldFont       = *-bold,
    ItalicFont     = *-italic,
    BoldItalicFont = *-bolditalic,
    Scale          = MatchLowercase,
  ]{texgyrecursor}
\else
  \setmainfont{Times New Roman}
  \setsansfont{Arial}
  \ifmcm@mac
    \setmonofont[Scale=MatchLowercase]{Menlo}
  \else
    \setmonofont[Scale=MatchLowercase]{Courier New}
  \fi
\fi

\ifmcm@mac
  \setCJKmainfont[
       UprightFont = * Light,
          BoldFont = * Bold,
        ItalicFont = Kaiti SC,
    BoldItalicFont = Kaiti SC Bold,
  ]{Songti SC}
  \setCJKsansfont{Heiti SC}
  \setCJKfamilyfont{zhsong}[
       UprightFont = * Light,
          BoldFont = * Bold,
  ]{Songti SC}
  \setCJKfamilyfont{zhhei}{Heiti SC}
  \setCJKfamilyfont{zhkai}{Kaiti SC}
  \setCJKfamilyfont{zhli}{Baoli SC}
  \newcommand*\lishu{\CJKfamily{zhli}}
\else
  \ifmcm@windows
    \IfFileExists{C:/bootfont.bin}{
      \setCJKmainfont[AutoFakeBold=2.15,ItalicFont=KaiTi_GB2312]{SimSun}
      \setCJKfamilyfont{zhkai}[AutoFakeBold=2.15]{KaiTi_GB2312}
    }{
      \setCJKmainfont[AutoFakeBold=2.15,ItalicFont=KaiTi]{SimSun}
      \setCJKfamilyfont{zhkai}[AutoFakeBold=2.15]{KaiTi}
    }
    \setCJKsansfont[AutoFakeBold=2.15]{SimHei}
    \setCJKfamilyfont{zhsong}[AutoFakeBold=2.15]{SimSun}
    \setCJKfamilyfont{zhhei}[AutoFakeBold=2.15]{SimHei}
    \ifx\lishu\undefined%
      \setCJKfamilyfont{zhli}{LiSu}
      \newcommand*\lishu{\CJKfamily{zhli}} %% Do this if it is undefined
    \fi
  \else
    \ifmcm@fandol
      % \setCJKfamilyfont{zhli}{LiSu} % 需要隶书字体 最好更换其他相似字体
      % \newcommand*\lishu{\CJKfamily{zhli}}
      \providecommand{\lishu}{\kaishu}
      \PackageWarning{heavenlyclouds}{用楷书代替隶书}
    \fi
  \fi
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 定理类环境（普通样式：加粗标签 + 小四宋体正文，符合官方格式规范）
\RequirePackage{amsthm}
\newtheoremstyle{gmcmplain}%
  {6pt plus 2pt minus 2pt}% 上方间距
  {6pt plus 2pt minus 2pt}% 下方间距
  {\normalfont}% 正文字体（宋体小四）
  {}% 缩进
  {\bfseries}% 标题字体
  {}% 标题后标点
  {0.5em}% 标题与正文间距
  {}% 标题格式：定义 1（标题）
\theoremstyle{gmcmplain}
\newtheorem{definition}{定义}
\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{proposition}{命题}
\newtheorem{assumption}{假设}
\newtheorem{conjecture}{猜想}
\newtheorem{axiom}{公理}
\newtheorem{principle}{定律}
\newtheorem{problem}{问题}
\newtheorem{example}{例}
% 不带编号的星号版本
\newtheorem*{definition*}{定义}
\newtheorem*{theorem*}{定理}
\newtheorem*{lemma*}{引理}
\newtheorem*{corollary*}{推论}
\newtheorem*{proposition*}{命题}
\newtheorem*{assumption*}{假设}
\newtheorem*{conjecture*}{猜想}
\newtheorem*{problem*}{问题}
\newtheorem*{example*}{例}

% 证明环境（自动补全证明结束符号；末行为公式时可用 \qedhere）
\renewenvironment{proof}[1][证明]{%
  \par\pushQED{\qed}\normalfont
  \topsep6\p@\@plus6\p@\relax
  \trivlist
  \item[\hskip\labelsep{\bfseries #1}]\ignorespaces
}{\popQED\endtrivlist\@endpefalse}
\renewcommand{\qedsymbol}{$\square$}
% 解环境
\newenvironment{solution}[1][解]{%
  \par\noindent{\bfseries #1}\quad\ignorespaces
}{\par\ignorespacesafterend}
% 注记环境
\newenvironment{note}[1][注]{%
  \par\noindent{\bfseries #1}\quad\ignorespaces
}{\par\ignorespacesafterend}

% 提示/要点框（灰色，黑白打印无差异）
% 用法：\begin{keybox}[标题] ... \end{keybox}
% 适合放「本文主要工作」「模型求解思路」「结论小结」等
\newtcolorbox{keybox}[1][提示]{%
  enhanced, breakable,
  colframe=gray!55, colback=gray!4, colbacktitle=gray!10,
  coltitle=black, fonttitle={\bfseries},
  boxrule=0.4pt, arc=1pt,
  before skip=10pt plus 2pt minus 2pt,
  after skip=10pt plus 2pt minus 2pt,
  top=6pt, bottom=6pt, left=8pt, right=8pt,
  title={#1},
}

% 带圈序号：\circled{1} \circled{2}
\newcommand{\circled}[1]{%
  \tikz[baseline=(gmcm@circled.base)]{%
    \node[shape=circle,draw,inner sep=0pt,minimum size=1.4em,font=\small]
      (gmcm@circled) {#1};}}

% 引用简写（\algref 已由 algorithmicx 宏包定义，用于引用算法行号，
% 引用整个算法请直接写 算法~\ref{alg:xxx}）
\newcommand{\figref}[1]{图~\ref{#1}}
\newcommand{\tabref}[1]{表~\ref{#1}}
\newcommand{\eqnref}[1]{式~\eqref{#1}}

% \autoref 中文名称
\renewcommand{\figureautorefname}{图}
\renewcommand{\tableautorefname}{表}
\renewcommand{\equationautorefname}{式}
\renewcommand{\appendixautorefname}{附录}
\@ifundefined{algorithmautorefname}%
  {\newcommand{\algorithmautorefname}{算法}}%
  {\renewcommand{\algorithmautorefname}{算法}}
\@ifundefined{subfigureautorefname}%
  {\newcommand{\subfigureautorefname}{图}}%
  {\renewcommand{\subfigureautorefname}{图}}

% URL 与正文同字体（参考文献里的网址更协调）
\urlstyle{same}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 代码环境（listings）
%% 基本用法：\begin{Python}{mip1.py} ... \end{Python}
%% bwprint 模式使用灰度高亮（打印友好），colorprint 模式使用彩色高亮
\if@gmcm@bwprint
  \definecolor{gmcm@keyword}{RGB}{0,0,0}
  \definecolor{gmcm@string}{RGB}{80,80,80}
  \definecolor{gmcm@comment}{RGB}{120,120,120}
\else
  \definecolor{gmcm@keyword}{RGB}{0,0,255}
  \definecolor{gmcm@string}{RGB}{160,32,240}
  \definecolor{gmcm@comment}{RGB}{34,139,34}
\fi

\lstdefinestyle{gmcm@codestyle}{
  xleftmargin=10pt,
  xrightmargin=10pt,
  frame=single,
  framesep=4pt,
  framerule=0.4pt,
  basicstyle=\small\ttfamily\linespread{1.0}\selectfont,
  keywordstyle=\color{gmcm@keyword}\bfseries,
  stringstyle=\color{gmcm@string},
  commentstyle=\color{gmcm@comment}\itshape,
  backgroundcolor=\color{gray!5},
  rulecolor=\color{gray!50},
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=4,
  captionpos=t,
  breaklines=true,
}

\renewcommand{\lstlistingname}{程序}

\lstnewenvironment{Matlab}[1]{%
  \lstset{style=gmcm@codestyle, language=matlab, title={#1},
  keywords={break,case,catch,classdef,continue,else,elseif,end,for,
  function,global,if,otherwise,parfor,persistent,return,spmd,switch,try,while}}}{}
\lstnewenvironment{Python}[1]{%
  \lstset{style=gmcm@codestyle, language=python, title={#1}}}{}
% 通用代码环境（其他语言或伪代码）
\lstnewenvironment{Code}[1]{%
  \lstset{style=gmcm@codestyle, title={#1}}}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 生成封面, 使用\maketitle
\renewcommand{\maketitle}{\par
	\begingroup
      \newpage
      \global\@topnum\z@ % Prevents figures from going at top of page.
      \ge@maketitle      %
    \endgroup
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\ge@maketitle{%
  \clearpage
  %\thispagestyle{plain}%
  \thispagestyle{empty}
  \pdfbookmark[1]{封~面}{cover}
  \if@gmcm@preface
  %\setcounter{page}{0}
  %\def\thepage{0}

 \begin{center}
 {\includegraphics{logo}}

  \vskip1.5cm
 {\zihao{2} \gmcm@ges@string@contents \par}
 \end{center}

 \vskip4em

 % \thispagestyle{gmcmheadings}
 \renewcommand\arraystretch{1.5}
 %\noindent
 \begin{tabularx}{\textwidth-40pt}{lX}
  \zihao{-2}\bfseries 学\qquad 校&\zihao{-2}\bfseries\gmcm@tokens@schoolname\\
  \Xhline{1.2pt}\\
  \zihao{-2}\bfseries 参赛队号&\zihao{-2}\bfseries\gmcm@tokens@baominghao\\
  \Xhline{1.2pt}\\[-15pt]
   \multirow{3}{*}{\zihao{-2}\bfseries 队员姓名}&\zihao{-2}\bfseries   1. \gmcm@tokens@membera\hfill\null\\
   \cline{2-2}\\[-15pt]
                               &\zihao{-2}\bfseries 2. \gmcm@tokens@memberb\hfill\null\\
   \cline{2-2}\\[-15pt]
                        &\zihao{-2}\bfseries 3.  \gmcm@tokens@memberc\hfill\null\\
   \Xhline{1.2pt}
  \end{tabularx}

	\makenametitle
	}

\def\makenametitle{
\clearpage
\thispagestyle{plain}%
\pdfbookmark[1]{标~题}{title}
\setcounter{page}{1}
\begin{center}
%{\includegraphics[width=4cm]{logo}}
{\zihao{2} \gmcm@ges@string@contents \par}
\end{center}

\vskip0.6cm%

\begin{center}
\begin{tabularx}{.95\textwidth}{@{}lX@{}}
{\zihao{-2}\lishu 题\quad 目: } &\hfill\zihao{3}\heiti \@title\hfill\null\\
\cline{2-2}
\end{tabularx}
\end{center}

\vskip0.2cm%
}


% 中文标题格式设置
% 目录深度
\setcounter{secnumdepth}{3}
\def\@seccntformat#1{\csname the#1\endcsname\ \ }
% 节标题格式, 居中, 使用\chinese命令修改计数器, \kern 使得数字和内容不至于太远
\newcommand{\thinskip}{\hskip 0.05em\relax}
% \renewcommand\thesection{\arabic{section}\thinskip.}
% 编号统一建立在 \thesection 之上：附录中自动变为 附录A、附录A.1……
\renewcommand\thesubsection{\thesection\thinskip.\thinskip\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection\thinskip.\thinskip\arabic{subsubsection}}
% 一级标题: 四号黑体居中（官方格式规范）
\renewcommand\section{\@startsection{section}{1}{\z@}%
 	{2.5ex \@plus -1ex \@minus -.2ex}%
	{2.3ex \@plus.2ex}%
	{\bfseries\centering\zihao{4}\heiti}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
	{1.25ex\@plus -1ex \@minus -.2ex}%
	{1.25ex \@plus .2ex}%
	{\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
	{1.25ex\@plus -1ex \@minus -.2ex}%
	{1.2ex \@plus .2ex}%
	{\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
	{3.25ex \@plus1ex \@minus.2ex}%
	{-1em}%
	{\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
	 {3.25ex \@plus1ex \@minus .2ex}%
	 {-1em}%
	 {\normalfont\normalsize\bfseries}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define new environment and command
%% 摘要两个字设置为 -2 号.
%% 定义摘要环境
\renewenvironment{abstract}{%
\if@twocolumn
		\section*{\gmcm@cap@abstractname}%
	\else
	 \begin{center}%
	{\zihao{-2}\lishu\gmcm@cap@abstractname\vspace{\z@}}%
	 \end{center}%
 	 \quotation
	\fi}
	{\if@twocolumn\else\endquotation\clearpage\fi}
\renewenvironment{quotation}
	{\list{}{\listparindent 2em%
	 \itemindent \listparindent
	 \rightmargin\z@
	 \leftmargin\z@
	 \parsep \z@ \@plus\p@}%
	\item\relax}
	{\endlist}

\newcommand\keywords[1]{%
	\renewcommand{\gmcm@tokens@keywords}{#1}
	\par
	\vskip1ex
	{\noindent\zihao{-3}\lishu\gmcm@cap@keywordsname：}~{\gmcm@tokens@keywords}
	}


\newcommand*\baominghao[1]{%
	\renewcommand{\gmcm@tokens@baominghao}{#1}}
\newcommand*\schoolname[1]{%
	\renewcommand{\gmcm@tokens@schoolname}{#1}}
\newcommand*\membera[1]{%
	\renewcommand{\gmcm@tokens@membera}{#1}}
\newcommand*\memberb[1]{%
	\renewcommand{\gmcm@tokens@memberb}{#1}}
\newcommand*\memberc[1]{%
	\renewcommand{\gmcm@tokens@memberc}{#1}}

% \title 同时写入 PDF 元数据
\renewcommand{\title}[1]{%
  \gdef\@title{#1}%
  \hypersetup{pdftitle={#1}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
 %%%  中文标题名称
\newcommand*{\gmcm@cap@abstractname}{摘\quad 要： }
 %%%  中文关键字
\newcommand*{\gmcm@cap@keywordsname}{关键词}

% 常量

%\newcommand\gmcm@ges@string@contents{“华为杯”第十五届全国研究生\par 数学建模竞赛}
\newcommand\gmcm@ges@string@contents{\includegraphics{title}}

\renewcommand{\thetable}{\thesection.\arabic{table}}
\renewcommand{\theequation}{\thesection.\arabic{equation}}
\renewcommand{\thefigure}{\thesection.\arabic{figure}}

% 字图与子标题
\RequirePackage[titles]{tocloft}
\renewcommand{\cftsecdotsep}{4.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\heiti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecaftersnumb}{\hskip.4em}

% ----------- 设置新的生成目录命令 -------------
\newcommand{\maketoc}{
  \phantomsection
  \pdfbookmark[1]{目~录}{contents}
  \tableofcontents
}

\AtBeginEnvironment{thebibliography}{%
    \phantomsection
    \addcontentsline{toc}{section}{\refname}
    }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 附录
%% 用法：\begin{appendices} \section{...} \end{appendices}
%% 编号自动处理：小节标题为 附录A、附录A.1，图表公式编号为 附录A.1 等，
%% 无需再手动 \renewcommand\thesubsection。
\newcommand*{\gmcm@cap@appendixname}{附录}
\renewcommand*\appendixname{\gmcm@cap@appendixname}

\renewcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \setcounter{subsubsection}{0}%
  \setcounter{equation}{0}%
  \setcounter{figure}{0}%
  \setcounter{table}{0}%
  \gdef\thesection{\appendixname\@Alph\c@section}%
  % hyperref 的 PDF 锚点需要无中文的计数形式，避免目标重复
  \gdef\theHsection{appendix.\@Alph\c@section}%
  \gdef\theHsubsection{appendix.\@Alph\c@section.\arabic{subsection}}%
  \gdef\theHsubsubsection{appendix.\@Alph\c@section.\arabic{subsection}.\arabic{subsubsection}}%
  \gdef\theHequation{appendix.\@Alph\c@section.\arabic{equation}}%
  \gdef\theHfigure{appendix.\@Alph\c@section.\arabic{figure}}%
  \gdef\theHtable{appendix.\@Alph\c@section.\arabic{table}}%
  % 目录中「附录A」比普通节号宽，需要加宽数字栏
  \addtocontents{toc}{\protect\renewcommand{\protect\cftsecnumwidth}{3.9em}}%
  \addtocontents{toc}{\protect\renewcommand{\protect\cftsubsecnumwidth}{5.8em}}%
  \addtocontents{toc}{\protect\renewcommand{\protect\cftsubsubsecnumwidth}{7.7em}}%
}

\newenvironment{appendices}{\appendix\ignorespaces}{\ignorespacesafterend}

% 浮动环境设置
% 默认情况下, \LaTeX{} 要求每页的文字至少占据 20%,否则该页就只单独放置一个浮动环境,
% 而这通常不是我们想要的, 我们将这个要求降低到 5%.
 \renewcommand*{\textfraction}{0.05}
% 有时如果多个浮动环境连续放在一起, \LaTeX{}
% 会将它们分在几个不同页，即使它们可在同一页放
% 得下. 我们可以通过修改 |\topfraction| 和 |\bottomfraction| 分别设置顶端和底端的浮
% 动环境的最大比例.
 \renewcommand*{\topfraction}{0.9}
 \renewcommand*{\bottomfraction}{0.8}
% 有时\LaTeX{}会把一个浮动环境单独放在一页,
% 我们要求这个环境至少要占据 85% 才能单独放在一页.
% 注意:  |\floatpagefraction| 的数值必须小于 |\topfraction|.
 \renewcommand*{\floatpagefraction}{0.85}
% 关于图片 graphicx
 % 如果图片没有指定后缀, 依次按下列顺序搜索
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
 % 设置图表搜索路径, 可以给图表文件夹取如下名字
\graphicspath{{figures/}{figure/}{pictures/}%
  {picture/}{pic/}{pics/}{image/}{images/}}
% 图表标题
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour}, % 图的字体, 宋体小四
	position=bottom % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}
\captionsetup[table]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour}, % 表的字体, 宋体小四
	position=top % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}
% 列表环境设置
\setlist{%
	topsep=0.3em, % 列表顶端的垂直空白
	partopsep=0pt, % 列表环境前面紧接着一个空白行时其顶端的额外垂直空白
	itemsep=0ex plus 0.1ex, % 列表项之间的额外垂直空白
	parsep=0pt, % 列表项内的段落之间的垂直空白
	leftmargin=1.5em, % 环境的左边界和列表之间的水平距离
	rightmargin=0em, % 环境的右边界和列表之间的水平距离
	labelsep=0.5em, % 包含标签的盒子与列表项的第一行文本之间的间隔
	labelwidth=2em % 包含标签的盒子的正常宽度；若实际宽度更宽，则使用实际宽度。
}

% \pagestyle{plain}

\endinput
