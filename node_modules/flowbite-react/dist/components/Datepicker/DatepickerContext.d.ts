import type { DatepickerTheme } from "./Datepicker";
import type { Views, WeekStart } from "./helpers";
export interface DatepickerContextValue {
    theme: DatepickerTheme;
    language: string;
    weekStart: WeekStart;
    minDate?: Date;
    maxDate?: Date;
    filterDate?: (date: Date, view: Views) => boolean;
    isOpen?: boolean;
    setIsOpen: (isOpen: boolean) => void;
    view: Views;
    setView: (value: Views) => void;
    selectedDate: Date | null;
    setSelectedDate: (date: Date | null) => void;
    changeSelectedDate: (date: Date, useAutohide: boolean) => void;
    viewDate: Date;
    setViewDate: (date: Date) => void;
}
export declare const DatepickerContext: import("react").Context<DatepickerContextValue | undefined>;
export declare function useDatePickerContext(): DatepickerContextValue;
