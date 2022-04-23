load "./taylor_diagram.ncl"

begin
  title = "2015-08-11_Ohashi"

  var = (/"01d15h", "02d15h", "03d15h", "05d15h", "07d15h", "10d15h", "20d15h", "30d15h", "Average"/)
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
  CA_ratio   = (/1.560892724084055,1.6696399071251142,1.6968926715947579,1.6892436783757065,1.5682782102548354,1.560892724084055,1.3831474969994184,1.5812840528937342,1.4621621938100755/)
  CA_cc      = (/0.7396974458162261,0.7553183600580987,0.7131729410932256,0.7366247149892863,0.7608527752135591,0.7396974458162261,0.5999268238320996,0.6176672628972529,0.7715052512427485/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"red"/)
  plot(0)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "ERA"
; "Wind Speed"
  CA_ratio   = (/1.680619263618401,1.4734834835758315,1.3934466215505101,1.3604053417549145,1.5453708522897802,1.3250773463160017,1.4086671832446238,0.7979670846795678,1.0569350703962266/)
  CA_cc      = (/0.7454724632436954,0.664654187347605,0.4293447274194841,0.40870794858332865,0.3685267513014649,0.5282549812072386,0.3378303560804386,0.34464493041302274,0.6394532871860322/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"blue"/)
  plot(1)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "GPV"
; "Wind Speed"
  CA_ratio   = (/1.4292485168112319,1.4677053976303907,1.5362736886754842,1.3313806793126564,1.3640468398165462,1.145318151730823,0.9684108649295384,1.4605063160766627,1.1015297383726959/)
  CA_cc      = (/0.7133133255176222,0.7874318959020538,0.7045283510576957,0.7883041953853204,0.724260005486787,0.4403949184476297,0.2692772114066426,0.8271337320234611,0.8247874557763025/)
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
