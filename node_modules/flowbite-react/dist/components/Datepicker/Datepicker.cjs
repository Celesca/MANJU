'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var arrowLeftIcon = require('../../icons/arrow-left-icon.cjs');
var arrowRightIcon = require('../../icons/arrow-right-icon.cjs');
var calendarIcon = require('../../icons/calendar-icon.cjs');
var provider = require('../../theme/provider.cjs');
var TextInput = require('../TextInput/TextInput.cjs');
var DatepickerContext = require('./DatepickerContext.cjs');
var helpers = require('./helpers.cjs');
var theme = require('./theme.cjs');
var Days = require('./Views/Days.cjs');
var Decades = require('./Views/Decades.cjs');
var Months = require('./Views/Months.cjs');
var Years = require('./Views/Years.cjs');

const Datepicker = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.datePickerTheme, provider$1.theme?.datepicker, props.theme],
    [get.get(provider$1.clearTheme, "datepicker"), props.clearTheme],
    [get.get(provider$1.applyTheme, "datepicker"), props.applyTheme]
  );
  const {
    title,
    open,
    inline = false,
    autoHide = true,
    // Hide when selected the day
    showClearButton = true,
    labelClearButton = "Clear",
    showTodayButton = true,
    labelTodayButton = "Today",
    defaultValue,
    minDate,
    maxDate,
    filterDate,
    language = "en",
    weekStart = helpers.WeekStart.Sunday,
    className,
    onChange,
    label,
    value,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.datepicker);
  const initialDate = defaultValue ? helpers.getFirstDateInRange(defaultValue, minDate, maxDate) : null;
  const effectiveDefaultView = React.useMemo(() => {
    return defaultValue ? helpers.getFirstDateInRange(defaultValue, minDate, maxDate) : /* @__PURE__ */ new Date();
  }, []);
  const [isOpen, setIsOpen] = React.useState(open);
  const [view, setView] = React.useState(helpers.Views.Days);
  const [selectedDate, setSelectedDate] = React.useState(value ?? initialDate);
  const [viewDate, setViewDate] = React.useState(value ?? effectiveDefaultView);
  const inputRef = React.useRef(null);
  const datepickerRef = React.useRef(null);
  function changeSelectedDate(date, useAutohide) {
    setSelectedDate(date);
    if ((date === null || date) && onChange) {
      onChange(date);
    }
    if (autoHide && view === helpers.Views.Days && useAutohide == true && !inline) {
      setIsOpen(false);
    }
  }
  function clearDate() {
    changeSelectedDate(initialDate, true);
    if (defaultValue) {
      setViewDate(defaultValue);
    }
  }
  React.useImperativeHandle(ref, () => ({
    focus() {
      inputRef.current?.focus();
    },
    clear() {
      clearDate();
    }
  }));
  function renderView(type) {
    switch (type) {
      case helpers.Views.Decades:
        return /* @__PURE__ */ jsxRuntime.jsx(Decades.DatepickerViewsDecades, {});
      case helpers.Views.Years:
        return /* @__PURE__ */ jsxRuntime.jsx(Years.DatepickerViewsYears, {});
      case helpers.Views.Months:
        return /* @__PURE__ */ jsxRuntime.jsx(Months.DatepickerViewsMonth, {});
      case helpers.Views.Days:
      default:
        return /* @__PURE__ */ jsxRuntime.jsx(Days.DatepickerViewsDays, {});
    }
  }
  function getNextView() {
    switch (view) {
      case helpers.Views.Days:
        return helpers.Views.Months;
      case helpers.Views.Months:
        return helpers.Views.Years;
      case helpers.Views.Years:
        return helpers.Views.Decades;
    }
    return view;
  }
  function getViewTitle() {
    switch (view) {
      case helpers.Views.Decades:
        return `${helpers.startOfYearPeriod(viewDate, 100) - 10} - ${helpers.startOfYearPeriod(viewDate, 100) + 100}`;
      case helpers.Views.Years:
        return `${helpers.startOfYearPeriod(viewDate, 10)} - ${helpers.startOfYearPeriod(viewDate, 10) + 11}`;
      case helpers.Views.Months:
        return helpers.getFormattedDate(language, viewDate, { year: "numeric" });
      case helpers.Views.Days:
      default:
        return helpers.getFormattedDate(language, viewDate, { month: "long", year: "numeric" });
    }
  }
  function getViewDatePage(view2, date, value2) {
    switch (view2) {
      case helpers.Views.Days:
        return new Date(helpers.addMonths(date, value2));
      case helpers.Views.Months:
        return new Date(helpers.addYears(date, value2));
      case helpers.Views.Years:
        return new Date(helpers.addYears(date, value2 * 10));
      case helpers.Views.Decades:
        return new Date(helpers.addYears(date, value2 * 100));
      default:
        return new Date(helpers.addYears(date, value2 * 10));
    }
  }
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      const clickedInsideDatepicker = datepickerRef?.current?.contains(event.target);
      const clickedInsideInput = inputRef?.current?.contains(event.target);
      if (!clickedInsideDatepicker && !clickedInsideInput) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [inputRef, datepickerRef, setIsOpen]);
  React.useEffect(() => {
    const effectiveValue = value && helpers.getFirstDateInRange(new Date(value), minDate, maxDate);
    const effectiveSelectedDate = selectedDate && helpers.getFirstDateInRange(new Date(selectedDate), minDate, maxDate);
    if (effectiveSelectedDate && effectiveValue && !helpers.isDateEqual(effectiveValue, effectiveSelectedDate)) {
      setSelectedDate(effectiveValue);
    }
    if (selectedDate == null) {
      setSelectedDate(initialDate);
    }
  }, [value, setSelectedDate, setViewDate, selectedDate]);
  const displayValue = value === null ? label : helpers.getFormattedDate(language, selectedDate || /* @__PURE__ */ new Date());
  return /* @__PURE__ */ jsxRuntime.jsx(
    DatepickerContext.DatepickerContext.Provider,
    {
      value: {
        theme: theme$1,
        language,
        minDate,
        maxDate,
        filterDate,
        weekStart,
        isOpen,
        setIsOpen,
        view,
        setView,
        viewDate,
        setViewDate,
        selectedDate,
        setSelectedDate,
        changeSelectedDate
      },
      children: /* @__PURE__ */ jsxRuntime.jsxs("div", { className: tailwindMerge.twMerge(theme$1.root.base, className), children: [
        !inline && /* @__PURE__ */ jsxRuntime.jsx(
          TextInput.TextInput,
          {
            theme: theme$1.root.input,
            icon: calendarIcon.CalendarIcon,
            ref: inputRef,
            onFocus: () => {
              if (selectedDate && !helpers.isDateEqual(viewDate, selectedDate)) {
                setViewDate(selectedDate);
              }
              setIsOpen(true);
            },
            value: displayValue,
            defaultValue: initialDate ? helpers.getFormattedDate(language, initialDate) : label,
            readOnly: true,
            ...restProps
          }
        ),
        (isOpen || inline) && /* @__PURE__ */ jsxRuntime.jsx("div", { ref: datepickerRef, className: tailwindMerge.twMerge(theme$1.popup.root.base, inline && theme$1.popup.root.inline), children: /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.popup.root.inner, children: [
          /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.popup.header.base, children: [
            title && /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.popup.header.title, children: title }),
            /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.popup.header.selectors.base, children: [
              /* @__PURE__ */ jsxRuntime.jsx(
                "button",
                {
                  type: "button",
                  className: tailwindMerge.twMerge(
                    theme$1.popup.header.selectors.button.base,
                    theme$1.popup.header.selectors.button.prev
                  ),
                  onClick: () => setViewDate(getViewDatePage(view, viewDate, -1)),
                  children: /* @__PURE__ */ jsxRuntime.jsx(arrowLeftIcon.ArrowLeftIcon, {})
                }
              ),
              /* @__PURE__ */ jsxRuntime.jsx(
                "button",
                {
                  type: "button",
                  className: tailwindMerge.twMerge(
                    theme$1.popup.header.selectors.button.base,
                    theme$1.popup.header.selectors.button.view
                  ),
                  onClick: () => setView(getNextView()),
                  children: getViewTitle()
                }
              ),
              /* @__PURE__ */ jsxRuntime.jsx(
                "button",
                {
                  type: "button",
                  className: tailwindMerge.twMerge(
                    theme$1.popup.header.selectors.button.base,
                    theme$1.popup.header.selectors.button.next
                  ),
                  onClick: () => setViewDate(getViewDatePage(view, viewDate, 1)),
                  children: /* @__PURE__ */ jsxRuntime.jsx(arrowRightIcon.ArrowRightIcon, {})
                }
              )
            ] })
          ] }),
          /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme$1.popup.view.base, children: renderView(view) }),
          (showClearButton || showTodayButton) && /* @__PURE__ */ jsxRuntime.jsxs("div", { className: theme$1.popup.footer.base, children: [
            showTodayButton && /* @__PURE__ */ jsxRuntime.jsx(
              "button",
              {
                type: "button",
                className: tailwindMerge.twMerge(theme$1.popup.footer.button.base, theme$1.popup.footer.button.today),
                onClick: () => {
                  const today = /* @__PURE__ */ new Date();
                  changeSelectedDate(today, true);
                  setViewDate(today);
                },
                children: labelTodayButton
              }
            ),
            showClearButton && /* @__PURE__ */ jsxRuntime.jsx(
              "button",
              {
                type: "button",
                className: tailwindMerge.twMerge(theme$1.popup.footer.button.base, theme$1.popup.footer.button.clear),
                onClick: () => {
                  changeSelectedDate(null, true);
                },
                children: labelClearButton
              }
            )
          ] })
        ] }) })
      ] })
    }
  );
});
Datepicker.displayName = "Datepicker";

exports.Datepicker = Datepicker;
//# sourceMappingURL=Datepicker.cjs.map
