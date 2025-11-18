import type { ThemingProps } from "../../types";
import type { DrawerTheme } from "./Drawer";
export interface DrawerContextValue extends ThemingProps<DrawerTheme> {
    id?: string;
    isOpen?: boolean;
    onClose?: () => void;
}
export declare const DrawerContext: import("react").Context<DrawerContextValue | undefined>;
export declare function useDrawerContext(): DrawerContextValue;
