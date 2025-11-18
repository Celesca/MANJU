import type { MutableRefObject } from "react";
import { type ComponentPropsWithoutRef } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbitePositions, FlowbiteSizes, ThemingProps } from "../../types";
import type { ModalBodyTheme } from "./ModalBody";
import type { ModalFooterTheme } from "./ModalFooter";
import type { ModalHeaderTheme } from "./ModalHeader";
export interface ModalTheme {
    root: ModalRootTheme;
    content: ModalContentTheme;
    body: ModalBodyTheme;
    header: ModalHeaderTheme;
    footer: ModalFooterTheme;
}
export interface ModalRootTheme {
    base: string;
    show: FlowbiteBoolean;
    sizes: ModalSizes;
    positions: ModalPositions;
}
export interface ModalContentTheme {
    base: string;
    inner: string;
}
export interface ModalPositions extends FlowbitePositions {
    [key: string]: string;
}
export interface ModalSizes extends Omit<FlowbiteSizes, "xs"> {
    [key: string]: string;
}
export interface ModalProps extends ComponentPropsWithoutRef<"div">, ThemingProps<ModalTheme> {
    onClose?: () => void;
    position?: DynamicStringEnumKeysOf<ModalPositions>;
    popup?: boolean;
    root?: HTMLElement;
    show?: boolean;
    size?: DynamicStringEnumKeysOf<ModalSizes>;
    dismissible?: boolean;
    initialFocus?: number | MutableRefObject<HTMLElement | null>;
}
export declare const Modal: import("react").ForwardRefExoticComponent<ModalProps & import("react").RefAttributes<HTMLDivElement>>;
