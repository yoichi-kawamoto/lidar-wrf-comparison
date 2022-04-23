load "./taylor_diagram.ncl"

begin
  title = "2015-09-11_Ohashi"

  var = (/"01d15h", "02d15h", "03d15h", "05d15h", "07d15h", "10d15h", "20d15h", "30d15h", "60d15h", "Average"/)
  nVar = dimsizes(var)

  case = (/"Wind Speed"/)
  nCase = dimsizes(case)

  wks = gsn_open_wks("pdf", title)
  plot = new(3, graphic)

  opt            = True
  opt@Markers    = (/4/)
  opt@gsMarkerSizeF = 0.006

  ; opt@varLabels = var
  opt@caseLabels = case

  opt@varLabelsYloc = 0.65
  opt@varLabelsFontHeightF = 0.02
  opt@caseLabelsFontHeightF = 0.1

  opt@stnRad = (/0.5, 1.5/)
  opt@ccRays = (/0.6, 0.9/)

  opt@centerDiffRMS = True

  opt@taylorDraw = False
  opt@taylorFrame = False

  opt@tiMainString = "FNL"
; "Wind Speed"
  CA_ratio   = (/1.0020250712537,1.1612903499019551,0.9265044976916584,1.5971920512044953,1.1216862121851918,1.1706920474116027,1.049055038215226,1.1722873075372686,1.0007477040151371,0.8679021989852967/)
  CA_cc      = (/0.8186176776141596,0.19403311050438835,0.5927478213093146,-0.08865357249489636,0.11184474701108377,-0.0010772367459692734,0.44207414380161264,0.13843090064876246,0.12806327492503633,0.2984955475275726/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"red"/)
  plot(0)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "ERA"
; "Wind Speed"
  CA_ratio   = (/1.0023249746265477,0.6399052064849664,0.7128023621120726,1.602166296734076,1.3874120247449229,1.1209371648087754,1.254751858295147,1.2835386437701182,1.219806077893066,0.9193455126070292/)
  CA_cc      = (/0.9096557699460892,0.3982320792151691,0.3640796309753691,-0.119636880752945,0.10576710069019907,0.26235464345004583,0.07395578573807392,0.10976301492552647,0.23383040149139955,0.2651855989392715/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"blue"/)
  plot(1)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "GPV"
; "Wind Speed"
  CA_ratio   = (/1.1604203398665351,0.8478829498225796,1.1288605030746095,2.2579675579449656,1.1190697405215113,1.3328043498640452,1.0221490924904622,1.2010426259425024,1.2508437068524405,0.9252646688937276/)
  CA_cc      = (/0.7473043492189685,0.4458714854484031,0.3563262552320314,-0.3556741201944147,-0.18664740610348127,-0.28193361142450574,-0.09091000862025174,-0.15977353500409916,-0.12237011042863062,-0.021392328415201414/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"green"/)
  plot(2)    = taylor_diagram(wks,ratio,cc,opt)

  resP = True
  resP@gsnPanelMainString = title
  resP@gsnPanelYWhiteSpacePercent = 3

  gsn_panel(wks, plot(0:2), (/1,3/), resP)

end
