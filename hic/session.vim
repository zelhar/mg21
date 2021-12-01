let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/my_gits/mg21/hic
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +399 matrixPermutationsDemonstrations00.py
badd +191 coolerStuff.py
badd +64 mymodule/hicCoolerModule.py
badd +11 mymodule/matrixpermutationsModule.py
badd +603 matrixpermutations04.py
argglobal
%argdel
$argadd matrixPermutationsDemonstrations00.py
tabnew
tabnew
tabnew
tabnew
tabrewind
edit matrixPermutationsDemonstrations00.py
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 399 - ((56 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 399
normal! 053|
lcd ~/my_gits/mg21/hic
tabnext
edit ~/my_gits/mg21/hic/matrixpermutations04.py
argglobal
balt ~/my_gits/mg21/hic/matrixPermutationsDemonstrations00.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 795 - ((62 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 795
normal! 016|
lcd ~/my_gits/mg21/hic
tabnext
edit ~/my_gits/mg21/hic/coolerStuff.py
argglobal
balt ~/my_gits/mg21/hic/matrixPermutationsDemonstrations00.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 158 - ((16 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 158
normal! 0
lcd ~/my_gits/mg21/hic
tabnext
edit ~/my_gits/mg21/hic/mymodule/hicCoolerModule.py
argglobal
balt ~/my_gits/mg21/hic/coolerStuff.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 15 - ((13 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 15
normal! 0
lcd ~/my_gits/mg21/hic/mymodule
tabnext
edit ~/my_gits/mg21/hic/mymodule/matrixpermutationsModule.py
argglobal
balt ~/my_gits/mg21/hic/mymodule/hicCoolerModule.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 400 - ((47 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 400
normal! 0
lcd ~/my_gits/mg21/hic/mymodule
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0&& getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOF
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
