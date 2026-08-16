import React from "react";

export interface ModalProps {
  className?: string;
}

export function Modal({ className }: ModalProps) {
  return (
    <div className={className}>
      <span>Modal placeholder</span>
    </div>
  );
}

export default Modal;
