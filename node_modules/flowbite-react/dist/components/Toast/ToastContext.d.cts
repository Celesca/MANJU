import type { ThemingProps } from "../../types";
import type { ToastDuration, ToastTheme } from "./Toast";
export interface ToastContextValue extends ThemingProps<ToastTheme> {
    duration?: ToastDuration;
    isClosed?: boolean;
    isRemoved?: boolean;
    setIsClosed: (isClosed: boolean) => void;
    setIsRemoved: (isRemoved: boolean) => void;
}
export declare const ToastContext: import("react").Context<ToastContextValue | undefined>;
export declare function useToastContext(): ToastContextValue;
