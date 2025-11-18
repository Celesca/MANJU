'use strict';

var theme$I = require('../components/Accordion/theme.cjs');
var theme$H = require('../components/Alert/theme.cjs');
var theme$G = require('../components/Avatar/theme.cjs');
var theme$F = require('../components/Badge/theme.cjs');
var theme$E = require('../components/Blockquote/theme.cjs');
var theme$D = require('../components/Breadcrumb/theme.cjs');
var theme$C = require('../components/Button/theme.cjs');
var theme$B = require('../components/Card/theme.cjs');
var theme$A = require('../components/Carousel/theme.cjs');
var theme$z = require('../components/Checkbox/theme.cjs');
var theme$y = require('../components/Clipboard/theme.cjs');
var theme$x = require('../components/DarkThemeToggle/theme.cjs');
var theme$w = require('../components/Datepicker/theme.cjs');
var theme$v = require('../components/Drawer/theme.cjs');
var theme$u = require('../components/Dropdown/theme.cjs');
var theme$t = require('../components/FileInput/theme.cjs');
var theme$s = require('../components/FloatingLabel/theme.cjs');
var theme$r = require('../components/Footer/theme.cjs');
var theme$q = require('../components/HelperText/theme.cjs');
var theme$p = require('../components/HR/theme.cjs');
var theme$o = require('../components/Kbd/theme.cjs');
var theme$n = require('../components/Label/theme.cjs');
var theme$m = require('../components/List/theme.cjs');
var theme$l = require('../components/ListGroup/theme.cjs');
var theme$k = require('../components/MegaMenu/theme.cjs');
var theme$j = require('../components/Modal/theme.cjs');
var theme$i = require('../components/Navbar/theme.cjs');
var theme$h = require('../components/Pagination/theme.cjs');
var theme$g = require('../components/Popover/theme.cjs');
var theme$f = require('../components/Progress/theme.cjs');
var theme$e = require('../components/Radio/theme.cjs');
var theme$d = require('../components/RangeSlider/theme.cjs');
var theme$c = require('../components/Rating/theme.cjs');
var theme$b = require('../components/Select/theme.cjs');
var theme$a = require('../components/Sidebar/theme.cjs');
var theme$9 = require('../components/Spinner/theme.cjs');
var theme$8 = require('../components/Table/theme.cjs');
var theme$7 = require('../components/Tabs/theme.cjs');
var theme$6 = require('../components/Textarea/theme.cjs');
var theme$5 = require('../components/TextInput/theme.cjs');
var theme$4 = require('../components/Timeline/theme.cjs');
var theme$3 = require('../components/Toast/theme.cjs');
var theme$2 = require('../components/ToggleSwitch/theme.cjs');
var theme$1 = require('../components/Tooltip/theme.cjs');
var config = require('./config.cjs');
var modeScript = require('./mode-script.cjs');
var provider = require('./provider.cjs');

const theme = {
  accordion: theme$I.accordionTheme,
  alert: theme$H.alertTheme,
  avatar: theme$G.avatarTheme,
  badge: theme$F.badgeTheme,
  blockquote: theme$E.blockquoteTheme,
  breadcrumb: theme$D.breadcrumbTheme,
  button: theme$C.buttonTheme,
  buttonGroup: theme$C.buttonGroupTheme,
  card: theme$B.cardTheme,
  carousel: theme$A.carouselTheme,
  checkbox: theme$z.checkboxTheme,
  clipboard: theme$y.clipboardTheme,
  darkThemeToggle: theme$x.darkThemeToggleTheme,
  datepicker: theme$w.datePickerTheme,
  drawer: theme$v.drawerTheme,
  dropdown: theme$u.dropdownTheme,
  fileInput: theme$t.fileInputTheme,
  floatingLabel: theme$s.floatingLabelTheme,
  footer: theme$r.footerTheme,
  helperText: theme$q.helperTextTheme,
  hr: theme$p.hrTheme,
  kbd: theme$o.kbdTheme,
  label: theme$n.labelTheme,
  list: theme$m.listTheme,
  listGroup: theme$l.listGroupTheme,
  megaMenu: theme$k.megaMenuTheme,
  modal: theme$j.modalTheme,
  navbar: theme$i.navbarTheme,
  pagination: theme$h.paginationTheme,
  popover: theme$g.popoverTheme,
  progress: theme$f.progressTheme,
  radio: theme$e.radioTheme,
  rangeSlider: theme$d.rangeSliderTheme,
  rating: theme$c.ratingTheme,
  ratingAdvanced: theme$c.ratingAdvancedTheme,
  select: theme$b.selectTheme,
  sidebar: theme$a.sidebarTheme,
  spinner: theme$9.spinnerTheme,
  table: theme$8.tableTheme,
  tabs: theme$7.tabsTheme,
  textarea: theme$6.textareaTheme,
  textInput: theme$5.textInputTheme,
  timeline: theme$4.timelineTheme,
  toast: theme$3.toastTheme,
  toggleSwitch: theme$2.toggleSwitchTheme,
  tooltip: theme$1.tooltipTheme
};

exports.ThemeConfig = config.ThemeConfig;
exports.ThemeModeScript = modeScript.ThemeModeScript;
exports.getThemeModeScript = modeScript.getThemeModeScript;
exports.initThemeMode = modeScript.initThemeMode;
exports.ThemeProvider = provider.ThemeProvider;
exports.useThemeProvider = provider.useThemeProvider;
exports.theme = theme;
//# sourceMappingURL=index.cjs.map
