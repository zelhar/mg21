" ~/.config/init.vim
"set nocompatible
" Remove ALL autocommands for the current group:
autocmd!

"make sure it stays on even if I delete Vundle or Neobunlde et al.
filetype plugin indent on

set termguicolors

"Switch on syntax highlighting if it wasn't on yet.
if !exists("syntax_on")
    syntax on
endif

"make backspace function like normal apps in insert mode
set backspace=indent,eol,start
set textwidth=80
set background=dark

"bracket highlight (that's on by default):
"set matchpairs=(:),{:},[:]
set showmatch

set expandtab
set smarttab
set softtabstop=4
set tabstop=4
set nu!
set shiftwidth=4

"set spelllang=en,de,es,he
"set nospell

" Read changes to file made by other applications
set autoread

"turn on auto-smart-indent
set autoindent
set smartindent


" Tell vim which characters to show for expanded TABs,
" trailing whitespace, and end-of-lines. VERY useful!
scriptencoding utf-8
if has("gui_running")
    set listchars=eol:¶,tab:>»,trail:·
"    set listchars=eol:¬,tab:>»,trail:·
else
"    set listchars=eol:¶,tab:>»,trail:·
    set listchars=eol:¬,tab:>»,trail:·
endif

" Show whitespace
"set list

set laststatus=2
set showtabline=2
"highlight all matches to search results
set hlsearch
" highlight match while still typing search pattern
set incsearch
" make default search ignore case
set ignorecase
" Set utf8 as standard encoding and en_US as the standard language
set encoding=utf8
" Use Unix as the standard file type
set ffs=unix,dos,mac

set ch=2		" Make command line two lines high
set cursorline
"set cc=81

" Setting scrolloff so cursor alsways stays inside that range except the top/bot
set scrolloff=5


" key-combination to move to next/previous buffer and tab (<C-l> same as redraw!)
"nnoremap <Leader>] :bn<CR>
"nnoremap <Leader>[ :bp<CR>
"nnoremap <Tab> :tabnext<Cr>
"nnoremap <C-Tab> :tabnext<Cr>
"nnoremap <S-Tab> :tabprevious<Cr>
"tnoremap <C-Tab> <C-\><C-n>:tabnext<Cr>

"splits the line after cursor and remain in normal mode
nnoremap <Leader><Enter> o<Esc>
nnoremap <M-Enter> i<Enter><Esc>

" can also simply use the unnamed register by default
"set clipboard+=unnamed

"make esc work as expected in neovim terminal:
tnoremap <Esc> <C-\><C-n>
tnoremap <A-h> <C-\><C-N><C-w>h
tnoremap <A-j> <C-\><C-N><C-w>j
tnoremap <A-k> <C-\><C-N><C-w>k
tnoremap <A-l> <C-\><C-N><C-w>l
inoremap <A-h> <C-\><C-N><C-w>h
inoremap <A-j> <C-\><C-N><C-w>j
inoremap <A-k> <C-\><C-N><C-w>k
inoremap <A-l> <C-\><C-N><C-w>l
nnoremap <A-h> <C-w>h
nnoremap <A-j> <C-w>j
nnoremap <A-k> <C-w>k
nnoremap <A-l> <C-w>l

"au BufNewFile,BufRead *.hs setlocal nospell

"Make vim save backup of the files: 
set backup
set backupcopy=auto
set backupdir=/run/media/zelhar/yjk-16g-msd/backupvimtexts/,
            \/run/media/zelhar/yjk-B16gb/backupvimtexts,
            \/run/media/zelhar/UF16/backupvimtexts,
            \/run/media/zelhar/JetFlash16/backupvimtexts,~/tmp,~/temp,.,~/,
            \/media/JetFlash16

"add a dictionary file for word completion (i_CTRL-X_CTRL-K):
"let g:symbols_file = "$HOME/dictionaries/symbols"
"set dictionary+=$HOME/dictionaries/symbols
"set dictionary+=$HOME/dictionaries/chemical_formulas.txt
set dictionary+=/usr/share/dict/american-english
set dictionary+=/usr/share/dict/ngerman
set dictionary+=/usr/share/dict/spanish
"make autocomplete (:help cpt) with ctrl-n search in the also in the dictionary
"set complete+=k
"set complete+=i
"set complete+=t
"set complete+=kspell
set completeopt=menuone,preview,longest,noinsert

"Set (locally) working dir to be the same as the file being edited in the buffer
autocmd BufEnter * silent! lcd %:p:h
"redraw screen when switching buffer, and returning to window (cleans garbage)  
"autocmd BufEnter * :redraw!
"autocmd FocusGained * :redraw! 
autocmd WinEnter * :filetype detect
"autocmd BufEnter * :filetype detect

" In many terminal emulators the mouse works just fine, thus enable it.
if has('mouse')
  set mouse=a
endif
set mousehide		" Hide the mouse when typing text

"set a shorter timeout for key-combs and commands (default=1000)
"set timeoutlen=1200
set timeoutlen=820
set showcmd
"set position for new split windows:
set splitbelow
set splitright


" Don't pass messages to |ins-completion-menu|.
set shortmess+=c
set signcolumn=yes
"set hidden
"set nohidden

set updatetime=300

"zelhar-backup
"defaults for my zelharbackup plugin:
let g:myfileslist = '/run/media/zelhar/yjk-16g-msd/original_paths_list.txt'
let g:mybackupdir=  '/run/media/zelhar/yjk-16g-msd/'
"call TurnOffZelharBackup()

